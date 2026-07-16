use std::{ffi::OsStr, sync::Arc, time::Instant};

use num_bigint::BigUint;
use rayon::prelude::*;
use rspack_collections::{IdentifierMap, IdentifierSet};
use rustc_hash::FxHashSet as HashSet;

use super::{
  BlockConnectionMap, BlockModules, CodeSplitter, ConnectionIdList, DependenciesBlockIdentifier,
  DependenciesBlockIdentifierMap, PreparedBlockConnectionMap, ProcessBlock, QueueAction,
  extract_block_modules,
};
use crate::{AsyncDependenciesBlockIdentifier, Compilation, ModuleIdentifier, RuntimeSpec};

const PARALLEL_CODE_SPLITTING_ENV: &str = "RSPACK_EXPERIMENTAL_PARALLEL_CODE_SPLITTING";
const PARALLEL_CODE_SPLITTING_STATS_ENV: &str = "RSPACK_EXPERIMENTAL_PARALLEL_CODE_SPLITTING_STATS";
const MIN_PARALLEL_CHUNK_GROUPS: usize = 2;

#[derive(Clone, Debug, Default)]
struct ParallelCodeSplitterStats {
  delayed_rounds: u32,
  delayed_actions: u32,
  parallel_batches: u32,
  parallel_jobs: u32,
  serial_actions: u32,
  stale_jobs: u32,
  processed_queue_items: u32,
  processed_blocks: u32,
  committed_modules: u32,
  global_pre_order_candidates: u32,
  global_post_order_candidates: u32,
  global_pre_order_checks: u32,
  global_post_order_checks: u32,
  local_cache_entries: u32,
  cached_block_entries: u32,
  max_batch_size: u32,
  walk_time_ns: u64,
  commit_time_ns: u64,
  cache_time_ns: u64,
  chunk_graph_time_ns: u64,
  chunk_group_order_time_ns: u64,
  global_order_time_ns: u64,
  async_blocks_time_ns: u64,
  skipped_items_time_ns: u64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ParallelCodeSplitterState {
  enabled: bool,
  stats_enabled: bool,
  stats: ParallelCodeSplitterStats,
  checked_pre_order_modules: IdentifierSet,
  checked_post_order_modules: IdentifierSet,
}

impl ParallelCodeSplitterState {
  pub(super) fn configure_from_env(&mut self) {
    self.enabled = std::env::var_os(PARALLEL_CODE_SPLITTING_ENV)
      .is_none_or(|value| value != OsStr::new("0") && !value.is_empty());
    self.stats_enabled = std::env::var_os(PARALLEL_CODE_SPLITTING_STATS_ENV)
      .is_some_and(|value| value != OsStr::new("0") && !value.is_empty());
    self.stats = Default::default();
    self.checked_pre_order_modules.clear();
    self.checked_post_order_modules.clear();
  }
}

#[derive(Debug)]
struct ParallelWalkJob {
  root: ProcessBlock,
  runtime: Arc<RuntimeSpec>,
  min_available_modules: Arc<BigUint>,
  chunk_mask: BigUint,
}

#[derive(Debug)]
enum WalkAction {
  ProcessBlock {
    block: DependenciesBlockIdentifier,
    module: ModuleIdentifier,
    queued: bool,
  },
  AddModule(ModuleIdentifier),
  LeaveModule(ModuleIdentifier),
}

#[derive(Debug)]
struct ParallelWalkResult {
  job: ParallelWalkJob,
  resulting_chunk_mask: BigUint,
  block_modules_cache: BlockConnectionMap,
  modules: Vec<ModuleIdentifier>,
  post_order_modules: Vec<ModuleIdentifier>,
  global_pre_order_candidates: Vec<ModuleIdentifier>,
  global_post_order_candidates: Vec<ModuleIdentifier>,
  async_blocks: Vec<(AsyncDependenciesBlockIdentifier, ModuleIdentifier)>,
  skipped_items: Vec<ModuleIdentifier>,
  skipped_connections: Vec<(ModuleIdentifier, ConnectionIdList)>,
  processed_queue_items: u32,
  processed_blocks: u32,
}

fn is_parallel_root(splitter: &CodeSplitter, root: &ProcessBlock) -> bool {
  let chunk_group_info = splitter.chunk_group_info(&root.chunk_group_info);
  chunk_group_info.chunk_loading && chunk_group_info.async_chunks
}

fn has_parallel_batch(splitter: &CodeSplitter) -> bool {
  let mut keys = HashSet::default();
  for action in &splitter.queue_delayed {
    let QueueAction::ProcessBlock(root) = action else {
      keys.clear();
      continue;
    };
    if !is_parallel_root(splitter, root) {
      keys.clear();
      continue;
    }

    let key = (root.chunk_group_info, root.chunk);
    if !keys.insert(key) {
      keys.clear();
      keys.insert(key);
    }
    if keys.len() >= MIN_PARALLEL_CHUNK_GROUPS {
      return true;
    }
  }
  false
}

/// Processes only the delayed roots produced by the legacy chunk-group merge loop. These roots
/// have already received their current `min_available_modules` and runtime. Keeping this boundary
/// avoids eagerly collecting work across connect/merge rounds, which is important for cycles and
/// named chunk groups.
pub(super) fn try_process_delayed_queue(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
) -> bool {
  if !splitter.parallel_state.enabled
    || rayon::current_num_threads() <= 1
    || splitter.queue_delayed.len() < MIN_PARALLEL_CHUNK_GROUPS
    || !has_parallel_batch(splitter)
  {
    return false;
  }

  let actions = std::mem::take(&mut splitter.queue_delayed);
  splitter.parallel_state.stats.delayed_rounds += 1;
  splitter.parallel_state.stats.delayed_actions += actions.len() as u32;

  let mut batch = Vec::new();
  let mut keys = HashSet::default();
  for action in actions {
    match action {
      QueueAction::ProcessBlock(root) if is_parallel_root(splitter, &root) => {
        let key = (root.chunk_group_info, root.chunk);
        if keys.contains(&key) {
          flush_batch(splitter, compilation, &mut batch);
          keys.clear();
        }
        keys.insert(key);
        batch.push(root);
      }
      action => {
        flush_batch(splitter, compilation, &mut batch);
        keys.clear();
        process_serial_action(splitter, compilation, action);
      }
    }
  }
  flush_batch(splitter, compilation, &mut batch);
  true
}

fn flush_batch(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  batch: &mut Vec<ProcessBlock>,
) {
  if batch.is_empty() {
    return;
  }
  if batch.len() < MIN_PARALLEL_CHUNK_GROUPS {
    splitter.parallel_state.stats.serial_actions += batch.len() as u32;
    for root in std::mem::take(batch) {
      process_serial_action(splitter, compilation, QueueAction::ProcessBlock(root));
    }
    return;
  }

  let roots = std::mem::take(batch);
  let stats = &mut splitter.parallel_state.stats;
  stats.parallel_batches += 1;
  stats.parallel_jobs += roots.len() as u32;
  stats.max_batch_size = stats.max_batch_size.max(roots.len() as u32);
  process_parallel_batch(splitter, compilation, roots);
}

fn process_serial_action(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  action: QueueAction,
) {
  debug_assert!(splitter.queue.is_empty());
  splitter.queue.push(action);
  splitter.process_queue(compilation);
  debug_assert!(splitter.queue.is_empty());
}

fn process_parallel_batch(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  roots: Vec<ProcessBlock>,
) {
  let jobs = roots
    .into_iter()
    .map(|root| {
      let chunk_group_info = splitter.chunk_group_info(&root.chunk_group_info);
      ParallelWalkJob {
        chunk_mask: splitter
          .mask_by_chunk
          .get(&root.chunk)
          .expect("chunk must be in mask_by_chunk")
          .clone(),
        runtime: chunk_group_info.runtime.clone(),
        min_available_modules: chunk_group_info.min_available_modules.clone(),
        root,
      }
    })
    .collect::<Vec<_>>();

  let prepared_blocks_map = &splitter.prepared_blocks_map;
  let prepared_connection_map = &splitter.prepared_connection_map;
  let ordinal_by_module = &splitter.ordinal_by_module;
  let block_modules_runtime_map = &splitter.block_modules_runtime_map;
  let measure = splitter.parallel_state.stats_enabled;
  let walk_start = measure.then(Instant::now);
  let results = jobs
    .into_par_iter()
    .map(|job| {
      let shared_block_modules = block_modules_runtime_map.get(&Some(job.runtime.clone()));
      walk_root(
        job,
        compilation,
        prepared_blocks_map,
        prepared_connection_map,
        ordinal_by_module,
        shared_block_modules,
      )
    })
    .collect::<Vec<_>>();

  if let Some(start) = walk_start {
    splitter.parallel_state.stats.walk_time_ns += start.elapsed().as_nanos() as u64;
  }
  splitter.parallel_state.stats.processed_queue_items += results
    .iter()
    .map(|result| result.processed_queue_items)
    .sum::<u32>();
  splitter.parallel_state.stats.processed_blocks += results
    .iter()
    .map(|result| result.processed_blocks)
    .sum::<u32>();
  splitter.parallel_state.stats.committed_modules += results
    .iter()
    .map(|result| result.modules.len() as u32)
    .sum::<u32>();

  let commit_start = measure.then(Instant::now);
  commit_results(splitter, compilation, results);
  if let Some(start) = commit_start {
    splitter.parallel_state.stats.commit_time_ns += start.elapsed().as_nanos() as u64;
  }
}

fn walk_root(
  job: ParallelWalkJob,
  compilation: &Compilation,
  prepared_blocks_map: &DependenciesBlockIdentifierMap<Vec<AsyncDependenciesBlockIdentifier>>,
  prepared_connection_map: &IdentifierMap<PreparedBlockConnectionMap>,
  ordinal_by_module: &IdentifierMap<u64>,
  shared_block_modules: Option<&BlockConnectionMap>,
) -> ParallelWalkResult {
  let mut actions = vec![WalkAction::ProcessBlock {
    block: job.root.block,
    module: job.root.module,
    queued: true,
  }];
  let mut chunk_mask = job.chunk_mask.clone();
  let mut block_modules_cache = BlockConnectionMap::default();
  let mut visited_blocks = HashSet::default();
  let mut modules = Vec::new();
  let mut post_order_modules = Vec::new();
  let mut async_blocks = Vec::new();
  let mut skipped_items = Vec::new();
  let mut skipped_connections = Vec::new();
  let mut processed_queue_items = 0;
  let mut processed_blocks = 0;

  while let Some(action) = actions.pop() {
    match action {
      WalkAction::ProcessBlock {
        block,
        module,
        queued,
      } => {
        if queued {
          processed_queue_items += 1;
        }
        processed_blocks += 1;
        if !visited_blocks.insert(block) {
          continue;
        }

        let block_modules = get_block_modules(
          block,
          &job.runtime,
          compilation,
          prepared_blocks_map,
          prepared_connection_map,
          shared_block_modules,
          &mut block_modules_cache,
        );
        for (target, active_state, connections) in block_modules.iter().rev() {
          let ordinal = *ordinal_by_module.get(target).unwrap_or_else(|| {
            panic!("expected a module ordinal for identifier '{target}', but none was found")
          });
          if chunk_mask.bit(ordinal) {
            continue;
          }
          if !active_state.is_true() {
            skipped_connections.push((*target, connections.clone()));
            if active_state.is_false() {
              continue;
            }
          }
          if active_state.is_true() && job.min_available_modules.bit(ordinal) {
            skipped_items.push(*target);
          } else if active_state.is_true() {
            actions.push(WalkAction::AddModule(*target));
          } else {
            actions.push(WalkAction::ProcessBlock {
              block: DependenciesBlockIdentifier::Module(*target),
              module,
              queued: true,
            });
          }
        }

        if let Some(blocks) = prepared_blocks_map.get(&block) {
          async_blocks.extend(blocks.iter().map(|block| (*block, module)));
        }
      }
      WalkAction::AddModule(module) => {
        processed_queue_items += 1;
        let ordinal = *ordinal_by_module.get(&module).unwrap_or_else(|| {
          panic!("expected a module ordinal for identifier '{module}', but none was found")
        });
        if chunk_mask.bit(ordinal) {
          continue;
        }
        if job.min_available_modules.bit(ordinal) {
          skipped_items.push(module);
          continue;
        }

        modules.push(module);
        chunk_mask.set_bit(ordinal, true);
        actions.push(WalkAction::LeaveModule(module));
        actions.push(WalkAction::ProcessBlock {
          block: DependenciesBlockIdentifier::Module(module),
          module,
          queued: false,
        });
      }
      WalkAction::LeaveModule(module) => {
        processed_queue_items += 1;
        post_order_modules.push(module);
      }
    }
  }

  // Global order indices are immutable while the workers are running. Do these graph lookups on
  // the worker threads and leave only the first-seen candidate checks for the ordered commit.
  let module_graph = compilation.get_module_graph();
  let global_pre_order_candidates = modules
    .iter()
    .filter(|module| {
      module_graph
        .module_graph_module_by_identifier(module)
        .is_none_or(|module| module.pre_order_index.is_none())
    })
    .copied()
    .collect();
  let global_post_order_candidates = post_order_modules
    .iter()
    .filter(|module| {
      module_graph
        .module_graph_module_by_identifier(module)
        .is_none_or(|module| module.post_order_index.is_none())
    })
    .copied()
    .collect();

  ParallelWalkResult {
    job,
    resulting_chunk_mask: chunk_mask,
    block_modules_cache,
    modules,
    post_order_modules,
    global_pre_order_candidates,
    global_post_order_candidates,
    async_blocks,
    skipped_items,
    skipped_connections,
    processed_queue_items,
    processed_blocks,
  }
}

fn get_block_modules(
  block: DependenciesBlockIdentifier,
  runtime: &Arc<RuntimeSpec>,
  compilation: &Compilation,
  prepared_blocks_map: &DependenciesBlockIdentifierMap<Vec<AsyncDependenciesBlockIdentifier>>,
  prepared_connection_map: &IdentifierMap<PreparedBlockConnectionMap>,
  shared_block_modules: Option<&BlockConnectionMap>,
  cache: &mut BlockConnectionMap,
) -> Arc<BlockModules> {
  if let Some(block_modules) = cache.get(&block) {
    return block_modules.clone();
  }
  if let Some(block_modules) = shared_block_modules.and_then(|cache| cache.get(&block)) {
    return block_modules.clone();
  }

  let root = block.get_root_block(compilation.get_module_graph());
  extract_block_modules(
    root,
    Some(runtime.clone()),
    compilation,
    prepared_blocks_map,
    prepared_connection_map,
    cache,
  );
  cache
    .get(&block)
    .cloned()
    .unwrap_or_else(|| Arc::new(Vec::new()))
}

fn result_is_stale(splitter: &CodeSplitter, result: &ParallelWalkResult) -> bool {
  let chunk_group_info = splitter.chunk_group_info(&result.job.root.chunk_group_info);
  if chunk_group_info.min_available_modules.as_ref() != result.job.min_available_modules.as_ref()
    || chunk_group_info.runtime.as_ref() != result.job.runtime.as_ref()
  {
    return true;
  }

  splitter
    .mask_by_chunk
    .get(&result.job.root.chunk)
    .expect("chunk must be in mask_by_chunk")
    != &result.job.chunk_mask
}

fn commit_results(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  results: Vec<ParallelWalkResult>,
) {
  let cache_start = splitter.parallel_state.stats_enabled.then(Instant::now);
  for result in &results {
    let (entries, inserted) = cache_block_modules(splitter, result);
    splitter.parallel_state.stats.local_cache_entries += entries;
    splitter.parallel_state.stats.cached_block_entries += inserted;
  }
  if let Some(start) = cache_start {
    splitter.parallel_state.stats.cache_time_ns += start.elapsed().as_nanos() as u64;
  }

  for result in results {
    if result_is_stale(splitter, &result) {
      splitter.parallel_state.stats.stale_jobs += 1;
      process_serial_action(
        splitter,
        compilation,
        QueueAction::ProcessBlock(result.job.root),
      );
      continue;
    }
    commit_result(splitter, compilation, result);
  }
}

fn cache_block_modules(splitter: &mut CodeSplitter, result: &ParallelWalkResult) -> (u32, u32) {
  let chunk_group_info = splitter.chunk_group_info(&result.job.root.chunk_group_info);
  if chunk_group_info.runtime.as_ref() != result.job.runtime.as_ref() {
    return (0, 0);
  }

  let runtime_cache = splitter
    .block_modules_runtime_map
    .entry(Some(result.job.runtime.clone()))
    .or_default();
  let mut inserted = 0;
  for (block, modules) in &result.block_modules_cache {
    if let std::collections::hash_map::Entry::Vacant(entry) = runtime_cache.entry(*block) {
      entry.insert(modules.clone());
      inserted += 1;
    }
  }
  (result.block_modules_cache.len() as u32, inserted)
}

fn commit_result(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  result: ParallelWalkResult,
) {
  let ParallelWalkResult {
    job,
    resulting_chunk_mask,
    modules,
    post_order_modules,
    global_pre_order_candidates,
    global_post_order_candidates,
    async_blocks,
    skipped_items,
    skipped_connections,
    processed_queue_items,
    processed_blocks,
    ..
  } = result;
  let chunk_group_info = job.root.chunk_group_info;
  let chunk = job.root.chunk;

  splitter.stat_processed_queue_items += processed_queue_items;
  splitter.stat_processed_blocks += processed_blocks;

  let measure = splitter.parallel_state.stats_enabled;
  splitter.parallel_state.stats.global_pre_order_candidates +=
    global_pre_order_candidates.len() as u32;
  splitter.parallel_state.stats.global_post_order_candidates +=
    global_post_order_candidates.len() as u32;

  let start = measure.then(Instant::now);
  compilation
    .build_chunk_graph_artifact
    .chunk_graph
    .connect_chunk_and_modules(chunk, &modules);
  *splitter
    .mask_by_chunk
    .get_mut(&chunk)
    .expect("chunk must be in mask_by_chunk") = resulting_chunk_mask;
  if let Some(start) = start {
    splitter.parallel_state.stats.chunk_graph_time_ns += start.elapsed().as_nanos() as u64;
  }

  let start = measure.then(Instant::now);
  let chunk_group_ukey = splitter.chunk_group_info(&chunk_group_info).chunk_group;
  {
    let chunk_group = compilation
      .build_chunk_graph_artifact
      .chunk_group_by_ukey
      .expect_get_mut(&chunk_group_ukey);
    chunk_group.module_pre_order_indices.reserve(modules.len());
    for module in &modules {
      if let std::collections::hash_map::Entry::Vacant(entry) =
        chunk_group.module_pre_order_indices.entry(*module)
      {
        entry.insert(chunk_group.next_pre_order_index);
        chunk_group.next_pre_order_index += 1;
      }
    }
    chunk_group
      .module_post_order_indices
      .reserve(post_order_modules.len());
    for module in &post_order_modules {
      if let std::collections::hash_map::Entry::Vacant(entry) =
        chunk_group.module_post_order_indices.entry(*module)
      {
        entry.insert(chunk_group.next_post_order_index);
        chunk_group.next_post_order_index += 1;
      }
    }
  }
  if let Some(start) = start {
    splitter.parallel_state.stats.chunk_group_order_time_ns += start.elapsed().as_nanos() as u64;
  }

  let start = measure.then(Instant::now);
  {
    let module_graph = compilation.get_module_graph_mut();
    for module in &global_pre_order_candidates {
      if !splitter
        .parallel_state
        .checked_pre_order_modules
        .insert(*module)
      {
        continue;
      }
      splitter.parallel_state.stats.global_pre_order_checks += 1;
      let module_graph_module = module_graph.module_graph_module_by_identifier_mut(module);
      if module_graph_module.pre_order_index.is_none() {
        module_graph_module.pre_order_index = Some(splitter.next_free_module_pre_order_index);
        splitter.next_free_module_pre_order_index += 1;
      }
    }
    for module in &global_post_order_candidates {
      if !splitter
        .parallel_state
        .checked_post_order_modules
        .insert(*module)
      {
        continue;
      }
      splitter.parallel_state.stats.global_post_order_checks += 1;
      let module_graph_module = module_graph.module_graph_module_by_identifier_mut(module);
      if module_graph_module.post_order_index.is_none() {
        module_graph_module.post_order_index = Some(splitter.next_free_module_post_order_index);
        splitter.next_free_module_post_order_index += 1;
      }
    }
  }
  if let Some(start) = start {
    splitter.parallel_state.stats.global_order_time_ns += start.elapsed().as_nanos() as u64;
  }

  let start = measure.then(Instant::now);
  for (block, module) in async_blocks {
    splitter.make_chunk_group(block, module, chunk_group_info, chunk, compilation);
  }
  if let Some(start) = start {
    splitter.parallel_state.stats.async_blocks_time_ns += start.elapsed().as_nanos() as u64;
  }

  let start = measure.then(Instant::now);
  let chunk_group_info = splitter.chunk_group_info_mut(&chunk_group_info);
  chunk_group_info.skipped_items.extend(skipped_items);
  chunk_group_info
    .skipped_module_connections
    .extend(skipped_connections);
  if let Some(start) = start {
    splitter.parallel_state.stats.skipped_items_time_ns += start.elapsed().as_nanos() as u64;
  }
}

pub(super) fn log_stats(splitter: &CodeSplitter) {
  if !splitter.parallel_state.stats_enabled {
    return;
  }
  let stats = &splitter.parallel_state.stats;
  eprintln!(
    "parallel-code-splitting delayed_rounds={} delayed_actions={} parallel_batches={} parallel_jobs={} serial_actions={} stale_jobs={} processed_queue_items={} processed_blocks={} committed_modules={} global_pre_candidates={} global_post_candidates={} global_pre_checks={} global_post_checks={} local_cache_entries={} cached_block_entries={} max_batch_size={} walk_ms={:.3} commit_ms={:.3} cache_ms={:.3} chunk_graph_ms={:.3} chunk_group_order_ms={:.3} global_order_ms={:.3} async_blocks_ms={:.3} skipped_items_ms={:.3}",
    stats.delayed_rounds,
    stats.delayed_actions,
    stats.parallel_batches,
    stats.parallel_jobs,
    stats.serial_actions,
    stats.stale_jobs,
    stats.processed_queue_items,
    stats.processed_blocks,
    stats.committed_modules,
    stats.global_pre_order_candidates,
    stats.global_post_order_candidates,
    stats.global_pre_order_checks,
    stats.global_post_order_checks,
    stats.local_cache_entries,
    stats.cached_block_entries,
    stats.max_batch_size,
    stats.walk_time_ns as f64 / 1_000_000.0,
    stats.commit_time_ns as f64 / 1_000_000.0,
    stats.cache_time_ns as f64 / 1_000_000.0,
    stats.chunk_graph_time_ns as f64 / 1_000_000.0,
    stats.chunk_group_order_time_ns as f64 / 1_000_000.0,
    stats.global_order_time_ns as f64 / 1_000_000.0,
    stats.async_blocks_time_ns as f64 / 1_000_000.0,
    stats.skipped_items_time_ns as f64 / 1_000_000.0,
  );
}
