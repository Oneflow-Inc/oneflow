#include "graph/build_graph.h"

namespace oneflow {

namespace {

std::unique_ptr<TaskGraph> BuildTaskGraphWithoutTransfm(
    const DLNetConf& dl_net_conf,
    const Strategy& strategy_conf,
    const IDMap& id_map,
    bool need_bp) {
  auto logical_graph = std::make_shared<LogicalGraph>();
  logical_graph->Init(dl_net_conf, strategy_conf);
  auto chain_graph = std::make_shared<ChainGraph>();
  chain_graph->Init(logical_graph);
  auto stage_graph = std::make_shared<StageGraph>();
  stage_graph->Init(chain_graph);
  std::unique_ptr<TaskGraph> task_graph(new TaskGraph);
  task_graph->Init(stage_graph, id_map, need_bp);
  return task_graph;
}

void BuildTransfmGraph4TaskGraph(
    TaskGraph* task_graph) {
  for (const auto& task_node : task_graph->nodes()) {
    task_node->SetNewTransfmGraph();
  }
  for (const auto& task_node : task_graph->nodes()) {
    if (task_node->IsFwNode()) {
      task_node->transfm_graph()->BuildGraph();
    }
  }
  for (const auto& task_node : task_graph->nodes()) {
    if (task_node->IsBpNode()) {
      task_node->transfm_graph()->BuildGraph();
    }
  }
  for (const auto& task_node : task_graph->nodes()) {
    task_node->transfm_graph()->SetupProducedRegisterDesc();
  }
  for (const auto& task_node : task_graph->nodes()) {
    task_node->transfm_graph()->SubscribeRegisterDescInnerPath();
  }
}

}

std::unique_ptr<TaskGraph> BuildTaskGraph(const DLNetConf& dl_net_conf,
                                          const Strategy& strategy_conf,
                                          const IDMap& id_map,
                                          bool need_bp) {
  auto task_graph = BuildTaskGraphWithoutTransfm(dl_net_conf,
                                                 strategy_conf,
                                                 id_map,
                                                 need_bp);
  BuildTransfmGraph4TaskGraph(task_graph.get());
  return task_graph;
}

} // namespace oneflow
