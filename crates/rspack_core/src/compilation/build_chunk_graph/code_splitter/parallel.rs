use std::{ffi::OsStr, sync::Arc};

use itertools::Itertools;
use num_bigint::BigUint;
use rayon::prelude::*;
use rspack_collections::{IdentifierIndexSet, IdentifierMap};
use rspack_util::fx_hash::{FxIndexMap, FxIndexSet};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::{
  BlockConnectionMap, BlockModules, CgiUkey, CodeSplitter, ConnectionIdList,
  DependenciesBlockIdentifier, DependenciesBlockIdentifierMap, PreparedBlockConnectionMap,
  ProcessBlock, get_active_state_of_connections,
};
use crate::{
  AsyncDependenciesBlockIdentifier, ChunkUkey, Compilation, ModuleIdentifier, RuntimeSpec,
};

const PARALLEL_CODE_SPLITTING_ENV: &str = "RSPACK_EXPERIMENTAL_PARALLEL_CODE_SPLITTING";
const MIN_PARALLEL_CHUNK_GROUPS: usize = 2;

#[derive(Clone, Debug, Default)]
pub(super) struct ParallelCodeSplitterState {
  enabled: bool,
  pending: FxIndexMap<CgiUkey, Vec<ProcessBlock>>,
  active_jobs: HashSet<CgiUkey>,
  revisions: HashMap<CgiUkey, u32>,
  // Runtime-specific connection states are immutable for one build_chunk_graph pass.
  prepared_runtimes: HashSet<Arc<RuntimeSpec>>,
}

impl ParallelCodeSplitterState {
  pub(super) fn configure_from_env(&mut self) {
    // Keep a process-level escape hatch while making the optimized path the default.
    self.enabled = std::env::var_os(PARALLEL_CODE_SPLITTING_ENV)
      .is_none_or(|value| value != OsStr::new("0") && !value.is_empty());
    if !self.enabled {
      self.pending.clear();
      self.active_jobs.clear();
      self.revisions.clear();
      self.prepared_runtimes.clear();
    }
  }

  pub(super) fn enabled(&self) -> bool {
    self.enabled
  }

  pub(super) fn has_pending_work(&self) -> bool {
    self.enabled && !self.pending.is_empty()
  }

  pub(super) fn enqueue(&mut self, cgi: CgiUkey, block: ProcessBlock) {
    self.pending.entry(cgi).or_default().push(block);
  }

  pub(super) fn bump_revision(&mut self, cgi: CgiUkey) {
    if self.active_jobs.contains(&cgi) {
      *self.revisions.entry(cgi).or_default() += 1;
    }
  }

  fn revision(&self, cgi: CgiUkey) -> u32 {
    self.revisions.get(&cgi).copied().unwrap_or_default()
  }
}

#[derive(Debug)]
struct ParallelWalkJob {
  cgi: CgiUkey,
  chunk: ChunkUkey,
  roots: Vec<ProcessBlock>,
  runtime: Arc<RuntimeSpec>,
  min_available_modules: Arc<BigUint>,
  chunk_mask: BigUint,
  revision: u32,
}

#[derive(Debug)]
enum WalkAction {
  ProcessBlock {
    block: DependenciesBlockIdentifier,
    module: ModuleIdentifier,
  },
  AddModule(ModuleIdentifier),
  LeaveModule(ModuleIdentifier),
}

#[derive(Debug)]
struct ParallelWalkResult {
  job: ParallelWalkJob,
  resulting_chunk_mask: BigUint,
  modules: Vec<ModuleIdentifier>,
  post_order_modules: Vec<ModuleIdentifier>,
  async_blocks: Vec<(AsyncDependenciesBlockIdentifier, ModuleIdentifier)>,
  skipped_items: IdentifierIndexSet,
  skipped_connections: FxIndexSet<(ModuleIdentifier, ConnectionIdList)>,
  processed_blocks: u32,
}

pub(super) fn process_parallel_work(splitter: &mut CodeSplitter, compilation: &mut Compilation) {
  let pending = std::mem::take(&mut splitter.parallel_state.pending);
  if pending.is_empty() {
    return;
  }
  if pending.len() < MIN_PARALLEL_CHUNK_GROUPS {
    // A narrow frontier cannot amortize Rayon scheduling or result allocation.
    for (_, roots) in pending {
      splitter
        .queue_delayed
        .extend(roots.into_iter().map(super::QueueAction::ProcessBlock));
    }
    return;
  }

  splitter
    .parallel_state
    .active_jobs
    .extend(pending.keys().copied());

  // Keep connection-state evaluation out of the worker hot loop. This is shared by all chunk
  // groups with the same runtime and is usually the highest-cardinality read-only preparation.
  prepare_runtime_block_modules(splitter, compilation, pending.keys().copied());

  let jobs = pending
    .into_iter()
    .map(|(cgi_ukey, roots)| {
      let cgi = splitter.chunk_group_info(&cgi_ukey);
      let chunk = roots
        .first()
        .expect("parallel code splitting job should have a root")
        .chunk;
      debug_assert!(roots.iter().all(|root| root.chunk == chunk));
      ParallelWalkJob {
        cgi: cgi_ukey,
        chunk,
        roots,
        runtime: cgi.runtime.clone(),
        min_available_modules: cgi.min_available_modules.clone(),
        chunk_mask: splitter
          .mask_by_chunk
          .get(&chunk)
          .expect("chunk must be in mask_by_chunk")
          .clone(),
        revision: splitter.parallel_state.revision(cgi_ukey),
      }
    })
    .collect_vec();

  let block_modules_runtime_map = &splitter.block_modules_runtime_map;
  let prepared_blocks_map = &splitter.prepared_blocks_map;
  let ordinal_by_module = &splitter.ordinal_by_module;
  let results = jobs
    .into_par_iter()
    .map(|job| {
      let block_modules = block_modules_runtime_map
        .get(&Some(job.runtime.clone()))
        .expect("parallel runtime block modules should be prepared");
      walk_chunk_group(job, block_modules, prepared_blocks_map, ordinal_by_module)
    })
    .collect::<Vec<_>>();

  let mut accepted_results = Vec::with_capacity(results.len());
  for mut result in results {
    // Earlier commits in this wave may have connected another block to the same named chunk group.
    // Such a result was computed from a stale availability/topology snapshot and must be retried.
    if result_is_stale(splitter, &result) {
      let cgi = result.job.cgi;
      for root in result.job.roots {
        splitter.parallel_state.enqueue(cgi, root);
      }
      continue;
    }

    commit_walk_topology(splitter, compilation, &mut result);
    accepted_results.push(result);
  }
  commit_walk_results(splitter, compilation, accepted_results);
  splitter.parallel_state.active_jobs.clear();
}

fn prepare_runtime_block_modules(
  splitter: &mut CodeSplitter,
  compilation: &Compilation,
  chunk_groups: impl Iterator<Item = CgiUkey>,
) {
  let runtimes = chunk_groups
    .map(|cgi| splitter.chunk_group_info(&cgi).runtime.clone())
    .unique()
    .filter(|runtime| !splitter.parallel_state.prepared_runtimes.contains(runtime))
    .collect_vec();

  for runtime in runtimes {
    let existing_block_modules = splitter
      .block_modules_runtime_map
      .get(&Some(runtime.clone()));
    let block_modules = splitter
      .prepared_connection_map
      .par_iter()
      .filter(|(module, _)| {
        existing_block_modules.is_none_or(|block_modules| {
          !block_modules.contains_key(&DependenciesBlockIdentifier::Module(**module))
        })
      })
      .fold(
        BlockConnectionMap::default,
        |mut block_modules, (module, connections)| {
          prepare_module_block_modules(
            *module,
            connections,
            runtime.as_ref(),
            compilation,
            &splitter.prepared_blocks_map,
            &mut block_modules,
          );
          block_modules
        },
      )
      .reduce(BlockConnectionMap::default, |mut left, right| {
        left.extend(right);
        left
      });

    let runtime_block_modules = splitter
      .block_modules_runtime_map
      .entry(Some(runtime.clone()))
      .or_default();
    runtime_block_modules.extend(block_modules);
    for block in splitter.prepared_blocks_map.keys() {
      runtime_block_modules
        .entry(*block)
        .or_insert_with(|| Arc::new(Vec::new()));
    }
    splitter.parallel_state.prepared_runtimes.insert(runtime);
  }
}

fn prepare_module_block_modules(
  module: ModuleIdentifier,
  connections: &PreparedBlockConnectionMap,
  runtime: &RuntimeSpec,
  compilation: &Compilation,
  prepared_blocks_map: &DependenciesBlockIdentifierMap<Vec<AsyncDependenciesBlockIdentifier>>,
  block_modules: &mut BlockConnectionMap,
) {
  let root = DependenciesBlockIdentifier::Module(module);
  let nested_blocks = prepared_blocks_map
    .get(&root)
    .map(Vec::as_slice)
    .unwrap_or_default();

  if nested_blocks.is_empty() {
    let modules = connections
      .iter()
      .map(|connection| {
        debug_assert_eq!(connection.block, root);
        let active_state = get_active_state_of_connections(
          &connection.connections,
          Some(runtime),
          compilation.get_module_graph(),
          &compilation.module_graph_cache_artifact,
          &compilation
            .build_module_graph_artifact
            .side_effects_state_artifact,
          &compilation.exports_info_artifact,
        );
        (
          connection.module,
          active_state,
          connection.connections.clone(),
        )
      })
      .collect();
    block_modules.insert(root, Arc::new(modules));
    return;
  }

  let mut modules_by_block =
    DependenciesBlockIdentifierMap::<BlockModules>::with_capacity_and_hasher(
      nested_blocks.len() + 1,
      Default::default(),
    );
  modules_by_block.insert(root, Vec::new());
  modules_by_block.extend(nested_blocks.iter().map(|block| {
    (
      DependenciesBlockIdentifier::AsyncDependenciesBlock(*block),
      Vec::new(),
    )
  }));

  for connection in connections {
    let active_state = get_active_state_of_connections(
      &connection.connections,
      Some(runtime),
      compilation.get_module_graph(),
      &compilation.module_graph_cache_artifact,
      &compilation
        .build_module_graph_artifact
        .side_effects_state_artifact,
      &compilation.exports_info_artifact,
    );
    modules_by_block.entry(connection.block).or_default().push((
      connection.module,
      active_state,
      connection.connections.clone(),
    ));
  }

  block_modules.extend(
    modules_by_block
      .into_iter()
      .map(|(block, modules)| (block, Arc::new(modules))),
  );
}

fn walk_chunk_group(
  job: ParallelWalkJob,
  block_modules: &BlockConnectionMap,
  prepared_blocks_map: &DependenciesBlockIdentifierMap<Vec<AsyncDependenciesBlockIdentifier>>,
  ordinal_by_module: &IdentifierMap<u64>,
) -> ParallelWalkResult {
  let mut actions = Vec::with_capacity(job.roots.len());
  for root in job.roots.iter().rev() {
    actions.push(WalkAction::ProcessBlock {
      block: root.block,
      module: root.module,
    });
  }

  let mut chunk_mask = job.chunk_mask.clone();
  let mut modules = Vec::new();
  let mut post_order_modules = Vec::new();
  let mut async_blocks = Vec::new();
  let mut skipped_items = IdentifierIndexSet::default();
  let mut skipped_connections = FxIndexSet::default();
  let mut visited_blocks = HashSet::default();
  let mut processed_blocks = 0;

  while let Some(action) = actions.pop() {
    match action {
      WalkAction::ProcessBlock { block, module } => {
        if !visited_blocks.insert(block) {
          continue;
        }
        processed_blocks += 1;
        if let Some(modules) = block_modules.get(&block) {
          for (target, active_state, connections) in modules.iter().rev() {
            let ordinal = *ordinal_by_module.get(target).unwrap_or_else(|| {
              panic!("expected a module ordinal for identifier '{target}', but none was found")
            });
            if chunk_mask.bit(ordinal) {
              continue;
            }
            if !active_state.is_true() {
              skipped_connections.insert((*target, connections.clone()));
              if active_state.is_false() {
                continue;
              }
            }
            if active_state.is_true() && job.min_available_modules.bit(ordinal) {
              skipped_items.insert(*target);
            } else if active_state.is_true() {
              actions.push(WalkAction::AddModule(*target));
            } else {
              actions.push(WalkAction::ProcessBlock {
                block: DependenciesBlockIdentifier::Module(*target),
                module,
              });
            }
          }
        }

        if let Some(blocks) = prepared_blocks_map.get(&block) {
          async_blocks.extend(blocks.iter().map(|block| (*block, module)));
        }
      }
      WalkAction::AddModule(module) => {
        let ordinal = *ordinal_by_module.get(&module).unwrap_or_else(|| {
          panic!("expected a module ordinal for identifier '{module}', but none was found")
        });
        if chunk_mask.bit(ordinal) {
          continue;
        }
        if job.min_available_modules.bit(ordinal) {
          skipped_items.insert(module);
          continue;
        }
        chunk_mask.set_bit(ordinal, true);
        modules.push(module);
        actions.push(WalkAction::LeaveModule(module));
        actions.push(WalkAction::ProcessBlock {
          block: DependenciesBlockIdentifier::Module(module),
          module,
        });
      }
      WalkAction::LeaveModule(module) => post_order_modules.push(module),
    }
  }

  ParallelWalkResult {
    job,
    resulting_chunk_mask: chunk_mask,
    modules,
    post_order_modules,
    async_blocks,
    skipped_items,
    skipped_connections,
    processed_blocks,
  }
}

fn result_is_stale(splitter: &CodeSplitter, result: &ParallelWalkResult) -> bool {
  let cgi = splitter.chunk_group_info(&result.job.cgi);
  cgi.min_available_modules.as_ref() != result.job.min_available_modules.as_ref()
    || cgi.runtime.as_ref() != result.job.runtime.as_ref()
    || splitter.parallel_state.revision(result.job.cgi) != result.job.revision
    || splitter
      .mask_by_chunk
      .get(&result.job.chunk)
      .expect("chunk must be in mask_by_chunk")
      != &result.job.chunk_mask
}

fn commit_walk_topology(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  result: &mut ParallelWalkResult,
) {
  splitter.stat_processed_blocks += result.processed_blocks;
  let cgi_ukey = result.job.cgi;
  let chunk = result.job.chunk;

  *splitter
    .mask_by_chunk
    .get_mut(&chunk)
    .expect("chunk must be in mask_by_chunk") = std::mem::take(&mut result.resulting_chunk_mask);

  for (block, module) in std::mem::take(&mut result.async_blocks) {
    splitter.make_chunk_group(block, module, cgi_ukey, chunk, compilation);
  }

  splitter.parallel_state.bump_revision(cgi_ukey);
}

fn commit_walk_results(
  splitter: &mut CodeSplitter,
  compilation: &mut Compilation,
  results: Vec<ParallelWalkResult>,
) {
  {
    let chunk_graph = &mut compilation.build_chunk_graph_artifact.chunk_graph;
    for result in &results {
      chunk_graph.connect_chunk_and_modules(result.job.chunk, &result.modules);
    }
  }

  {
    let chunk_group_by_ukey = &mut compilation.build_chunk_graph_artifact.chunk_group_by_ukey;
    for result in &results {
      let chunk_group_ukey = splitter.chunk_group_info(&result.job.cgi).chunk_group;
      let chunk_group = chunk_group_by_ukey.expect_get_mut(&chunk_group_ukey);
      chunk_group
        .module_pre_order_indices
        .reserve(result.modules.len());
      for module in &result.modules {
        if let std::collections::hash_map::Entry::Vacant(entry) =
          chunk_group.module_pre_order_indices.entry(*module)
        {
          entry.insert(chunk_group.next_pre_order_index);
          chunk_group.next_pre_order_index += 1;
        }
      }
      chunk_group
        .module_post_order_indices
        .reserve(result.post_order_modules.len());
      for module in &result.post_order_modules {
        if let std::collections::hash_map::Entry::Vacant(entry) =
          chunk_group.module_post_order_indices.entry(*module)
        {
          entry.insert(chunk_group.next_post_order_index);
          chunk_group.next_post_order_index += 1;
        }
      }
    }
  }

  {
    let module_graph = compilation.get_module_graph_mut();
    for result in &results {
      for module in &result.modules {
        let module_graph_module = module_graph.module_graph_module_by_identifier_mut(module);
        if module_graph_module.pre_order_index.is_none() {
          module_graph_module.pre_order_index = Some(splitter.next_free_module_pre_order_index);
          splitter.next_free_module_pre_order_index += 1;
        }
      }
      for module in &result.post_order_modules {
        let module_graph_module = module_graph.module_graph_module_by_identifier_mut(module);
        if module_graph_module.post_order_index.is_none() {
          module_graph_module.post_order_index = Some(splitter.next_free_module_post_order_index);
          splitter.next_free_module_post_order_index += 1;
        }
      }
    }
  }

  for result in results {
    let cgi = splitter.chunk_group_info_mut(&result.job.cgi);
    cgi.skipped_items.extend(result.skipped_items);
    cgi
      .skipped_module_connections
      .extend(result.skipped_connections);
  }
}
