#include <unordered_map>
#include <unordered_set>
#include "oneflow/core/compiler/of2xla/xla_node.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

extern const std::string _XlaLaunchOpType = "XlaLaunch";
extern const std::string _XlaArgumentOpType = "XlaArgument";
extern const std::string _XlaLaunchPrefix = "_xla_launch_";
extern const std::string _XlaInArgumentPrefix = "_input_argument_";
extern const std::string _XlaOutArgumentPrefix = "_output_argument_";

class CreateXlaLaunchOpPass : public XlaOptimizePass {
 public:
  CreateXlaLaunchOpPass(const OptimizeOptions &options)
      : XlaOptimizePass(options) {}

  void Run() override;

 private:
  void RebuildSubgraphInputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes);

  void RebuildSubgraphOutputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes);
};

void CreateXlaLaunchOpPass::Run() {
  XlaGraph *graph = this->optimize_options_.graph;

  std::unordered_map<int64_t, XlaNode *> launch_nodes;
  std::unordered_map<int64_t, std::unordered_set<XlaNode *>> folded_nodes;

  // Create xla launch nodes,
  // and redirect all outer inputs and outputs of the launch node
  for (XlaNode *node : graph->Nodes()) {
    int64_t cluster_id = node->cluster_id();
    if (cluster_id != -1) {
      XlaNode *launch_node = nullptr;
      if (launch_nodes.count(cluster_id) == 0) {
        launch_node = graph->AddNode();
        launch_nodes.emplace(cluster_id, launch_node);
      } else {
        launch_node = launch_nodes[cluster_id];
      }

      launch_node->set_cluster_id(cluster_id);
      launch_node->set_op_type(_XlaLaunchOpType);
      launch_node->set_backend(node->backend());
      // Assign the `op_name_` of the node, and it will be used to find the
      // `XlaLaunchConf` when fixing launch blob name
      launch_node->set_op_name(absl::StrCat(_XlaLaunchPrefix, cluster_id));

      // Redirect inputs and outputs of the launch node
      for (XlaEdge *e : node->in_edges()) {
        XlaNode *start = e->start();
        if (start->cluster_id() != cluster_id) {
          e->UpdateEndNode(launch_node);
          launch_node->AddInEdge(e);
        }
      }
      for (XlaEdge *e : node->out_edges()) {
        XlaNode *end = e->end();
        if (end->cluster_id() != cluster_id) {
          e->UpdateStartNode(launch_node);
          launch_node->AddOutEdge(e);
        }
      }

      folded_nodes[cluster_id].insert(node);
    }
  }

  // Build subgraph for xla launch nodes and repair error connections
  // caused by redirect. Add argument nodes and create connections
  // between them and folded nodes.
  for (auto &kv : folded_nodes) {
    int64_t cluster_id = kv.first;
    XlaNode *launch_node = launch_nodes[cluster_id];
    XlaGraph *sub_graph = graph->AddSubGraph(launch_node->unique_id());

    std::unordered_map<int64_t, XlaNode *> sub_graph_nodes;
    for (XlaNode *n : kv.second) {
      XlaNode *node = sub_graph->AddNode(n->node());
      sub_graph_nodes[n->unique_id()] = node;
      
      // Rebuild inputs if the end node of input edges has been changed,
      // otherwise repair input to the node of the subgraph
      RebuildSubgraphInputs(node, n, sub_graph, &sub_graph_nodes);

      // Rebuild outputs same as rebuilding the inputs
      RebuildSubgraphOutputs(node, n, sub_graph, &sub_graph_nodes);
    }
  }
}

void CreateXlaLaunchOpPass::RebuildSubgraphInputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes) {
  int argument_id = 0;
  for (XlaEdge *e : n->in_edges()) {
    int64_t start_id = e->start()->unique_id();
    if (e->end()->unique_id() != n->unique_id()) {
      XlaNode *argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddNode();
        argument->set_op_type(_XlaArgumentOpType);
        argument->set_backend(e->start()->backend());
        argument->set_op_name(absl::StrCat(_XlaInArgumentPrefix, argument_id));
        argument_id++;
        sub_graph_nodes->emplace(start_id, argument);
      } else {
        argument = (*sub_graph_nodes)[start_id];
      }
      sub_graph->Connect(argument, node, e->argument());
    } else {
      if (sub_graph_nodes->count(start_id) != 0) {
        XlaNode *start = (*sub_graph_nodes)[start_id];
        sub_graph->Connect(start, node, e->argument());
      }
    }
  }
}

void CreateXlaLaunchOpPass::RebuildSubgraphOutputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes) {
  int argument_id = 0;
  for (XlaEdge *e : n->out_edges()) {
    int64_t start_id = e->start()->unique_id();
    if (e->start()->unique_id() != node->unique_id()) {
      XlaNode *argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddNode();
        argument->set_op_type(_XlaArgumentOpType);
        argument->set_backend(e->start()->backend());
        argument->set_op_name(absl::StrCat(_XlaOutArgumentPrefix, argument_id));
        argument_id++;
        sub_graph_nodes->emplace(start_id, argument);
      } else {
        argument = (*sub_graph_nodes)[start_id];
      }
      sub_graph->Connect(node, argument, e->argument());
    } else {
      if (sub_graph_nodes->count(start_id) != 0) {
        XlaNode *end = (*sub_graph_nodes)[start_id];
         sub_graph->Connect(node, end, e->argument());
      }
    }
  }
}

REGISTER_OPTIMIZE_PASS(CreateXlaLaunchOp, CreateXlaLaunchOpPass);

}  // namespace mola
}  // namespace oneflow
