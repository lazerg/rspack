use std::{ffi::OsStr, sync::Arc};

use num_bigint::BigUint;
use rayon::prelude::*;
use rspack_collections::IdentifierMap;
use rustc_hash::FxHashSet as HashSet;

use super::{
  AddAndEnterModule, BlockConnectionMap, BlockModules, CodeSplitter, ConnectionIdList,
  DependenciesBlockIdentifier, DependenciesBlockIdentifierMap, PreparedBlockConnectionMap,
  ProcessBlock, QueueAction, extract_block_modules,
};
use crate::{
  AsyncDependenciesBlockIdentifier, ChunkUkey, Compilation, ModuleIdentifier, RuntimeSpec,
};

const PARALLEL_CODE_SPLITTING_ENV: &str = "RSPACK_EXPERIMENTAL_PARALLEL_CODE_SPLITTING";
const MIN_PARALLEL_ACTIONS: usize = 2;
const MIN_ESTIMATED_CONNECTIONS: usize = 64;
const MIN_ESTIMATED_CONNECTIONS_PER_ACTION: usize = 16;

#[derive(Clone, Debug, Default)]
pub(super) struct ParallelCodeSplitterState {
  enabled: bool,
}

impl ParallelCodeSplitterState {
  pub(super) fn configure_from_env(&mut self) {
    self.enabled = std::env::var_os(PARALLEL_CODE_SPLITTING_ENV)
      .is_none_or(|value| value != OsStr::new("0") && !value.is_empty());
  }
}

#[derive(Debug, Clone)]
enum ParallelRoot {
  AddModule(AddAndEnterModule),
  ProcessBlock(ProcessBlock),
}

impl ParallelRoot {
  fn from_queue_action(action: QueueAction) -> Self {
    match action {
      QueueAction::AddAndEnterModule(item) => Self::AddModule(item),
      QueueAction::ProcessBlock(item) => Self::ProcessBlock(item),
      _ => unreachable!("parallel roots only contain module and block actions"),
    }
  }

  fn chunk_group_info(&self) -> super::CgiUkey {
    match self {
      Self::AddModule(item) => item.chunk_group_info,
      Self::ProcessBlock(item) => item.chunk_group_info,
    }
  }

  fn chunk(&self) -> ChunkUkey {
    match self {
      Self::AddModule(item) => item.chunk,
      Self::ProcessBlock(item) => item.chunk,
    }
  }

  fn into_queue_action(self) -> QueueAction {
    match self {
      Self::AddModule(item) => QueueAction::AddAndEnterModule(item),
      Self::ProcessBlock(item) => QueueAction::ProcessBlock(item),
    }
  }

  fn to_walk_action(&self) -> WalkAction {
    match self {
      Self::AddModule(item) => WalkAction::AddModule(item.module),
      Self::ProcessBlock(item) => WalkAction::ProcessBlock {
        block: item.block,
        module: item.module,
        queued: true,
      },
    }
  }
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
struct ParallelWalkJob {
  root: ParallelRoot,
  runtime: Arc<RuntimeSpec>,
  min_available_modules: Arc<BigUint>,
  chunk_mask: BigUint,
}

#[derive(Debug)]
struct ParallelWalkResult {
  job: ParallelWalkJob,
  block_modules_cache: BlockConnectionMap,
  modules: Vec<ModuleIdentifier>,
  post_order_modules: Vec<ModuleIdentifier>,
  async_blocks: Vec<(AsyncDependenciesBlockIdentifier, ModuleIdentifier)>,
  skipped_items: Vec<ModuleIdentifier>,
  skipped_connections: Vec<(ModuleIdentifier, ConnectionIdList)>,
  checked_modules: Vec<ModuleIdentifier>,
  processed_queue_items: u32,
  processed_blocks: u32,
}

pub(super) fn should_process_discovered_actions(
  splitter: &CodeSplitter,
  actions: &[QueueAction],
) -> bool {
  if !splitter.parallel_state.enabled
    || rayon::current_num_threads() <= 1
    || actions.len() < MIN_PARALLEL_ACTIONS
  {
    return false;
  }

  let Some(first) = actions.first() else {
    return false;
  };
  let first = match first {
    QueueAction::AddAndEnterModule(item) => (item.chunk_group_info, item.chunk),
    QueueAction::ProcessBlock(item) => (item.chunk_group_info, item.chunk),
    _ => return false,
  };
  let chunk_group_info = splitter.chunk_group_info(&first.0);
  if !chunk_group_info.chunk_loading || !chunk_group_info.async_chunks {
    return false;
  }

  let mut estimated_connections = 0;
  for action in actions {
    let root = match action {
      QueueAction::AddAndEnterModule(item) => {
        if (item.chunk_group_info, item.chunk) != first {
          return false;
        }
        item.module
      }
      QueueAction::ProcessBlock(item) => {
        if (item.chunk_group_info, item.chunk) != first {
          return false;
        }
        match item.block {
          DependenciesBlockIdentifier::Module(module) => module,
          DependenciesBlockIdentifier::AsyncDependenciesBlock(_) => item.module,
        }
      }
      _ => return false,
    };
    let root_estimated_connections = splitter
      .prepared_connection_map
      .get(&root)
      .map_or(0, Vec::len);
    if root_estimated_connections < MIN_ESTIMATED_CONNECTIONS_PER_ACTION {
      return false;
    }
    estimated_connections += root_estimated_connections;
  }
  // A few large roots must not pull a wide-but-shallow batch into Rayon. Require enough total
  // work and enough independently useful work in every speculative root.
  estimated_connections >= MIN_ESTIMATED_CONNECTIONS
}

pub(super) fn process_discovered_actions(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  actions: Vec<QueueAction>,
) {
  let mut roots = actions
    .into_iter()
    .map(ParallelRoot::from_queue_action)
    .collect::<Vec<_>>();
  // The legacy queue is a stack. Actions were collected in push order, so reverse them to obtain
  // the exact order in which the legacy algorithm would process them.
  roots.reverse();

  let jobs = roots
    .into_iter()
    .map(|root| {
      let chunk_group_info = splitter.chunk_group_info(&root.chunk_group_info());
      ParallelWalkJob {
        chunk_mask: splitter
          .mask_by_chunk
          .get(&root.chunk())
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
  let shared_block_modules = jobs.first().and_then(|job| {
    splitter
      .block_modules_runtime_map
      .get(&Some(job.runtime.clone()))
  });
  let results = jobs
    .into_par_iter()
    .map(|job| {
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

  commit_results(splitter, compilation, results);
}

fn walk_root(
  job: ParallelWalkJob,
  compilation: &Compilation,
  prepared_blocks_map: &DependenciesBlockIdentifierMap<Vec<AsyncDependenciesBlockIdentifier>>,
  prepared_connection_map: &IdentifierMap<PreparedBlockConnectionMap>,
  ordinal_by_module: &IdentifierMap<u64>,
  shared_block_modules: Option<&BlockConnectionMap>,
) -> ParallelWalkResult {
  let mut actions = vec![job.root.to_walk_action()];
  let mut chunk_mask = job.chunk_mask.clone();
  let mut block_modules_cache = BlockConnectionMap::default();
  let mut visited_blocks = HashSet::default();
  let mut modules = Vec::new();
  let mut post_order_modules = Vec::new();
  let mut async_blocks = Vec::new();
  let mut skipped_items = Vec::new();
  let mut skipped_connections = Vec::new();
  let mut checked_modules = Vec::new();
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
          checked_modules.push(*target);
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
        checked_modules.push(module);
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

  ParallelWalkResult {
    job,
    block_modules_cache,
    modules,
    post_order_modules,
    async_blocks,
    skipped_items,
    skipped_connections,
    checked_modules,
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
  let chunk_group_info = splitter.chunk_group_info(&result.job.root.chunk_group_info());
  if chunk_group_info.min_available_modules.as_ref() != result.job.min_available_modules.as_ref()
    || chunk_group_info.runtime.as_ref() != result.job.runtime.as_ref()
  {
    return true;
  }

  let chunk_mask = splitter
    .mask_by_chunk
    .get(&result.job.root.chunk())
    .expect("chunk must be in mask_by_chunk");
  result.checked_modules.iter().any(|module| {
    let ordinal = splitter.ordinal_by_module.get(module).unwrap_or_else(|| {
      panic!("expected a module ordinal for identifier '{module}', but none was found")
    });
    chunk_mask.bit(*ordinal)
  })
}

fn commit_results(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  results: Vec<ParallelWalkResult>,
) {
  for result in &results {
    cache_block_modules(splitter, result);
  }

  let mut results = results.into_iter();
  while let Some(result) = results.next() {
    if result_is_stale(splitter, &result) {
      let mut roots = vec![result.job.root];
      roots.extend(results.map(|result| result.job.root));
      splitter
        .queue
        .extend(roots.into_iter().rev().map(ParallelRoot::into_queue_action));
      return;
    }
    commit_result(splitter, compilation, result);
  }
}

fn cache_block_modules(splitter: &mut CodeSplitter, result: &ParallelWalkResult) {
  let chunk_group_info = splitter.chunk_group_info(&result.job.root.chunk_group_info());
  if chunk_group_info.runtime.as_ref() != result.job.runtime.as_ref() {
    return;
  }

  let runtime_cache = splitter
    .block_modules_runtime_map
    .entry(Some(result.job.runtime.clone()))
    .or_default();
  for (block, modules) in &result.block_modules_cache {
    runtime_cache
      .entry(*block)
      .or_insert_with(|| modules.clone());
  }
}

fn commit_result(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  result: ParallelWalkResult,
) {
  let ParallelWalkResult {
    job,
    modules,
    post_order_modules,
    async_blocks,
    skipped_items,
    skipped_connections,
    processed_queue_items,
    processed_blocks,
    ..
  } = result;
  let chunk_group_info = job.root.chunk_group_info();
  let chunk = job.root.chunk();

  splitter.stat_processed_queue_items += processed_queue_items;
  splitter.stat_processed_blocks += processed_blocks;

  compilation
    .build_chunk_graph_artifact
    .chunk_graph
    .connect_chunk_and_modules(chunk, &modules);
  let chunk_mask = splitter
    .mask_by_chunk
    .get_mut(&chunk)
    .expect("chunk must be in mask_by_chunk");
  for module in &modules {
    let ordinal = splitter.ordinal_by_module.get(module).unwrap_or_else(|| {
      panic!("expected a module ordinal for identifier '{module}', but none was found")
    });
    chunk_mask.set_bit(*ordinal, true);
  }

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

  {
    let module_graph = compilation.get_module_graph_mut();
    for module in &modules {
      let module_graph_module = module_graph.module_graph_module_by_identifier_mut(module);
      if module_graph_module.pre_order_index.is_none() {
        module_graph_module.pre_order_index = Some(splitter.next_free_module_pre_order_index);
        splitter.next_free_module_pre_order_index += 1;
      }
    }
    for module in &post_order_modules {
      let module_graph_module = module_graph.module_graph_module_by_identifier_mut(module);
      if module_graph_module.post_order_index.is_none() {
        module_graph_module.post_order_index = Some(splitter.next_free_module_post_order_index);
        splitter.next_free_module_post_order_index += 1;
      }
    }
  }

  for (block, module) in async_blocks {
    splitter.make_chunk_group(block, module, chunk_group_info, chunk, compilation);
  }

  let chunk_group_info = splitter.chunk_group_info_mut(&chunk_group_info);
  chunk_group_info.skipped_items.extend(skipped_items);
  chunk_group_info
    .skipped_module_connections
    .extend(skipped_connections);
}
