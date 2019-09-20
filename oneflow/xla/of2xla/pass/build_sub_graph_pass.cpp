#include <unordered_map>
#include <unordered_set>
#include "absl/strings/str_cat.h"
#include "oneflow/xla/of2xla/xla_node.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/pass/xla_optimize_pass.h"

namespace oneflow {
namespace mola {

extern const std::string _XlaLaunchOpType = "XlaLaunch";
extern const std::string _XlaArgumentOpType = "XlaArgument";
extern const std::string _XlaLaunchPrefix = "_xla_launch_";
extern const std::string _XlaInArgumentPrefix = "_input_argument_";
extern const std::string _XlaOutArgumentPrefix = "_output_argument_";

class BuildSubGraphPass : public XlaOptimizePass {
 public:
  BuildSubGraphPass(const OptimizeOptions &options)
      : XlaOptimizePass(options) {}

  void Run() override;

 private:
  void RebuildSubgraphInputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes);

  void RebuildSubgraphOutputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes);

  void CreateLaunchNodes(XlaGraph *graph,
                         std::unordered_map<int64_t, XlaNode *> *launch_nodes);

  void DivideArgumentNodes(XlaGraph *sub_graph);
};

void BuildSubGraphPass::Run() {
  XlaGraph *graph = this->optimize_options_.graph;
  // Create xla launch nodes
  std::unordered_map<int64_t, XlaNode *> launch_nodes;
  CreateLaunchNodes(graph, &launch_nodes);

  // Redirect all outer edges of the launch nodes
  std::unordered_map<int64_t, std::unordered_set<XlaNode *>> folded_nodes;
  for (XlaNode *node : graph->Nodes()) {
    int64_t cluster_id = node->cluster_id();
    if (cluster_id != -1 && node->op_type() != _XlaLaunchOpType) {
      XlaNode *launch_node = launch_nodes[cluster_id];
      // Redirect input edges
      for (XlaEdge *e : node->in_edges()) {
        XlaNode *start = e->start();
        if (start->cluster_id() != cluster_id) {
          e->UpdateEndNode(launch_node);
          launch_node->AddInEdge(e);
        }
      }
      // Redirect output edges
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
  // between them and the folded nodes
  for (auto &kv : folded_nodes) {
    int64_t cluster_id = kv.first;
    XlaNode *launch_node = launch_nodes[cluster_id];
    XlaGraph *sub_graph = graph->AddSubGraph(launch_node->unique_id());

    std::unordered_map<int64_t, XlaNode *> sub_graph_nodes;
    for (XlaNode *n : kv.second) {
      XlaNode *node = sub_graph->AddNode(n->op());
      sub_graph_nodes[n->unique_id()] = node;
      
      // Rebuild inputs if the end node of input edges has been changed,
      // otherwise repair input for the node of the subgraph
      RebuildSubgraphInputs(node, n, sub_graph, &sub_graph_nodes);

      // Rebuild outputs same as rebuilding the inputs
      RebuildSubgraphOutputs(node, n, sub_graph, &sub_graph_nodes);
    }
    // Divide argument nodes if they have multiple inputs or outputs with
    // different argument (or `LogicalBlobId`), and then fill their `op_name_`
    DivideArgumentNodes(sub_graph);
  }

  for (const XlaNode *node : graph->Nodes()) {
    CHECK(!node->IsReachable(*node));
  }
}

void BuildSubGraphPass::CreateLaunchNodes(
    XlaGraph *graph, std::unordered_map<int64_t, XlaNode *> *launch_nodes) {
  std::unordered_map<int64_t, std::string> cluster_ids;
  for (XlaNode *node : graph->Nodes()) {
    int64_t cluster_id = node->cluster_id();
    if (cluster_id != -1) {
      cluster_ids.emplace(cluster_id, node->backend());
    }
  }
  
  for (const auto &pair : cluster_ids) {
    int64_t cluster_id = pair.first;
    XlaNode *launch_node = graph->AddNode();
    launch_node->set_cluster_id(cluster_id);
    launch_node->set_backend(pair.second);
    launch_node->set_op_type(_XlaLaunchOpType);

    // Assign the `op_name_` of the node, and it will be used to find the
    // `XlaLaunchConf` when fixing launch blob name
    launch_node->set_op_name(absl::StrCat(_XlaLaunchPrefix, cluster_id));

    launch_nodes->emplace(cluster_id, launch_node);
  }
}

void BuildSubGraphPass::DivideArgumentNodes(XlaGraph *sub_graph) {
  // Find all argument nodes
  std::vector<XlaNode *> argument_nodes;
  for (XlaNode *node : sub_graph->Nodes()) {
    if (node->op_type() == _XlaArgumentOpType) {
      argument_nodes.push_back(node);
    }
  }
  // Start to divide nodes
  int argument_id = 0;
  for (XlaNode *node : argument_nodes) {
    std::list<XlaEdge *> in_edges = node->in_edges();
    std::list<XlaEdge *> out_edges = node->out_edges();
    // Argument node should has either inputs or outputs
    CHECK(in_edges.size() == 0 || out_edges.size() == 0);

    node->ClearInEdges();
    node->ClearOutEdges();

    std::unordered_map<Argument, XlaNode *> divided_args;
    for (XlaEdge *edge : in_edges) {
      const Argument &arg = edge->argument();
      if (node->in_edges().size() == 0) {
        node->set_op_name(absl::StrCat(_XlaOutArgumentPrefix, argument_id++));
        divided_args.emplace(arg, node);
      }
      const auto &it = divided_args.find(arg);
      if (it == divided_args.end()) {
        XlaNode *argument = sub_graph->AddNode();
        argument->set_op_type(_XlaArgumentOpType);
        argument->set_op_name(absl::StrCat(_XlaOutArgumentPrefix,
                                           argument_id++));
        argument->set_backend(node->backend());
        argument->AddInEdge(edge);
        edge->UpdateEndNode(argument);
        divided_args.emplace(arg, argument);
      } else {
        XlaNode *consumer = it->second;
        edge->UpdateEndNode(consumer);
        consumer->AddInEdge(edge);
      }
    }

    for (XlaEdge *edge : out_edges) {
      const Argument &arg = edge->argument();
      if (node->out_edges().size() == 0) {
        node->set_op_name(absl::StrCat(_XlaInArgumentPrefix, argument_id++));
        divided_args.emplace(arg, node);
      }
      const auto &it = divided_args.find(arg);
      if (it == divided_args.end()) {
        XlaNode *argument = sub_graph->AddNode();
        argument->set_op_type(_XlaArgumentOpType);
        argument->set_op_name(absl::StrCat(_XlaInArgumentPrefix,
                                           argument_id++));
        argument->set_backend(node->backend());
        argument->AddOutEdge(edge);
        edge->UpdateStartNode(argument);
        divided_args.emplace(arg, argument);
      } else {
        XlaNode *producer = it->second;
        edge->UpdateStartNode(producer);
        producer->AddOutEdge(edge);
      }
    }
  }
}

void BuildSubGraphPass::RebuildSubgraphInputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes) {
  for (XlaEdge *e : n->in_edges()) {
    int64_t start_id = e->start()->unique_id();
    if (e->end()->unique_id() != n->unique_id()) {
      XlaNode *argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddNode();
        argument->set_op_type(_XlaArgumentOpType);
        argument->set_backend(e->start()->backend());
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

void BuildSubGraphPass::RebuildSubgraphOutputs(
    XlaNode *node, XlaNode *n, XlaGraph *sub_graph,
    std::unordered_map<int64_t, XlaNode *> *sub_graph_nodes) {
  for (XlaEdge *e : n->out_edges()) {
    int64_t start_id = e->start()->unique_id();
    if (e->start()->unique_id() != node->unique_id()) {
      XlaNode *argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddNode();
        argument->set_op_type(_XlaArgumentOpType);
        argument->set_backend(e->start()->backend());
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

REGISTER_OPTIMIZE_PASS(BuildSubGraph, BuildSubGraphPass);

}  // namespace mola
}  // namespace oneflow
