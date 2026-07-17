// use rspack_core::Bundle;
// use rspack_core::ChunkGraph;

use tracing::instrument;

use self::code_splitter::CodeSplitterStatsPhase;
use crate::Compilation;
pub(crate) mod code_splitter;
pub(crate) mod incremental;
pub(crate) mod pass;

#[instrument("Compilation:build_chunk_graph", skip_all)]
pub fn build_chunk_graph(compilation: &mut Compilation) -> rspack_error::Result<()> {
  // TODO: heuristic incremental update is temporarily disabled
  // Original code:
  // let enable_incremental = compilation
  //   .incremental
  //   .mutations_readable(IncrementalPasses::BUILD_CHUNK_GRAPH);
  let enable_incremental = false;
  let mut splitter = if enable_incremental {
    std::mem::take(&mut compilation.build_chunk_graph_artifact.code_splitter)
  } else {
    Default::default()
  };
  splitter.configure_parallel_stats();
  let total_start = splitter.stats_start();

  let start = splitter.stats_start();
  let all_modules = compilation
    .get_module_graph()
    .modules_keys()
    .copied()
    .collect::<Vec<_>>();
  splitter.record_stats_phase(CodeSplitterStatsPhase::CollectModules, start);

  splitter.prepare(&all_modules, compilation)?;

  let start = splitter.stats_start();
  splitter.update_with_compilation(compilation)?;
  splitter.record_stats_phase(CodeSplitterStatsPhase::UpdateWithCompilation, start);

  if !enable_incremental || splitter.chunk_group_infos.is_empty() {
    let start = splitter.stats_start();
    let inputs = splitter.prepare_input_entrypoints_and_modules(&all_modules, compilation)?;
    splitter.record_stats_phase(CodeSplitterStatsPhase::PrepareInput, start);
    let start = splitter.stats_start();
    splitter.prepare_entries(inputs, compilation)?;
    splitter.record_stats_phase(CodeSplitterStatsPhase::PrepareEntries, start);
  }

  splitter.split(compilation)?;

  // remove empty chunk groups
  let start = splitter.stats_start();
  splitter.remove_orphan(compilation)?;
  splitter.record_stats_phase(CodeSplitterStatsPhase::RemoveOrphan, start);

  // make sure all module (weak dependency particularly) has a cgm
  let start = splitter.stats_start();
  for module_identifier in all_modules {
    compilation
      .build_chunk_graph_artifact
      .chunk_graph
      .add_module(module_identifier)
  }
  splitter.record_stats_phase(CodeSplitterStatsPhase::EnsureModules, start);
  splitter.record_stats_phase(CodeSplitterStatsPhase::Total, total_start);
  splitter.log_parallel_stats();

  compilation
    .build_chunk_graph_artifact
    .set_code_splitter(splitter);

  Ok(())
}
