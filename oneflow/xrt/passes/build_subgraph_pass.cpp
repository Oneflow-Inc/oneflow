#include <fstream>
#include <iostream>

#include "absl/strings/str_cat.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/graph/node.h"
#include "oneflow/xrt/passes/pass.h"

namespace oneflow {
namespace xrt {

extern const std::string _XrtLaunchOpType = "XrtLaunch";
extern const std::string _ArgumentOpType = "Argument";
extern const std::string _XrtLaunchPrefix = "_xrt_launch_";
extern const std::string _XrtInArgumentPrefix = "_input_argument_";
extern const std::string _XrtOutArgumentPrefix = "_output_argument_";

class BuildSubGraphPass : public XrtPass {
 public:
  BuildSubGraphPass() = default;

  void Run(XrtGraph *graph, const XrtPassOptions &options) override;

 private:
  void RebuildSubgraphInputs(XrtNode *node, XrtNode *n, XrtGraph *sub_graph,
                             util::Map<int64_t, XrtNode *> *sub_graph_nodes);

  void RebuildSubgraphOutputs(XrtNode *node, XrtNode *n, XrtGraph *sub_graph,
                              util::Map<int64_t, XrtNode *> *sub_graph_nodes);

  void CreateLaunchNodes(XrtGraph *graph,
                         util::Map<int64_t, XrtNode *> *launch_nodes);

  int64_t ClusterId(const XrtNode *node) const {
    int64_t cluster_id = -1;
    if (node->HasAttr("cluster_id")) {
      cluster_id = node->Attr<int64_t>("cluster_id");
    }
    return cluster_id;
  }

  void DivideArgumentNodes(XrtGraph *sub_graph);
  void DumpSubgraphs(const XrtGraph *graph, const std::string &path);
};

void BuildSubGraphPass::Run(XrtGraph *graph, const XrtPassOptions &options) {
  CHECK(graph) << "Graph is required by `BuildSubGraphPass`.";
  // Create xrt launch nodes
  util::Map<int64_t, XrtNode *> launch_nodes;
  CreateLaunchNodes(graph, &launch_nodes);

  // Redirect all outer edges of the launch nodes
  util::Map<int64_t, util::Set<XrtNode *>> folded_nodes;
  for (XrtNode *node : graph->Nodes()) {
    int64_t cluster_id = ClusterId(node);
    if (cluster_id != -1 && node->type() != _XrtLaunchOpType) {
      XrtNode *launch_node = launch_nodes[cluster_id];
      // Redirect input edges
      for (XrtEdge *e : node->in_edges()) {
        XrtNode *start = e->start();
        if (ClusterId(start) != cluster_id) {
          e->SetEndNode(launch_node);
          launch_node->AddInEdge(e);
        }
      }
      // Redirect output edges
      for (XrtEdge *e : node->out_edges()) {
        XrtNode *end = e->end();
        if (ClusterId(end) != cluster_id) {
          e->SetStartNode(launch_node);
          launch_node->AddOutEdge(e);
        }
      }

      folded_nodes[cluster_id].insert(node);
    }
  }

  // Build subgraph for xrt launch nodes and repair error connections
  // caused by redirect. Add argument nodes and create connections
  // between them and the folded nodes
  for (auto &kv : folded_nodes) {
    int64_t cluster_id = kv.first;
    XrtNode *launch_node = launch_nodes[cluster_id];
    XrtGraph *sub_graph = graph->AddSubgraph(launch_node->unique_id());

    util::Map<int64_t, XrtNode *> sub_graph_nodes;
    for (XrtNode *n : kv.second) {
      XrtNode *node = sub_graph->AddNode(n->param());
      node->set_name(n->name());
      node->set_type(n->type());
      node->set_backend(n->backend());
      sub_graph_nodes[n->unique_id()] = node;

      // Rebuild inputs if the end node of input edges has been changed,
      // otherwise repair input for the node of the subgraph
      RebuildSubgraphInputs(node, n, sub_graph, &sub_graph_nodes);

      // Rebuild outputs same as rebuilding the inputs
      RebuildSubgraphOutputs(node, n, sub_graph, &sub_graph_nodes);
    }
    // Divide argument nodes if they have multiple inputs or outputs with
    // different argument (or `LogicalBlobId`), and then fill their `name_`
    DivideArgumentNodes(sub_graph);
  }

  for (const XrtNode *node : graph->Nodes()) {
    CHECK(!node->IsReachable(*node));
  }

  DumpSubgraphs(graph, "./dump_subgraph");
}

void BuildSubGraphPass::CreateLaunchNodes(
    XrtGraph *graph, util::Map<int64_t, XrtNode *> *launch_nodes) {
  util::Map<int64_t, XrtDevice> cluster_ids;
  for (XrtNode *node : graph->Nodes()) {
    int64_t cluster_id = ClusterId(node);
    if (cluster_id != -1) {
      cluster_ids.emplace(cluster_id, node->backend());
    }
  }

  for (const auto &pair : cluster_ids) {
    int64_t cluster_id = pair.first;
    XrtNode *launch_node = graph->AddNode();
    launch_node->SetAttr("cluster_id", cluster_id);
    launch_node->set_backend(pair.second);
    launch_node->set_type(_XrtLaunchOpType);

    // Assign the `name_` of the node, and it will be used to find the
    // `XrtLaunchConf` when fixing launch blob name
    launch_node->set_name(absl::StrCat(_XrtLaunchPrefix, cluster_id));

    launch_nodes->emplace(cluster_id, launch_node);
  }
}

void BuildSubGraphPass::DivideArgumentNodes(XrtGraph *sub_graph) {
  // Find all argument nodes
  std::vector<XrtNode *> argument_nodes;
  for (XrtNode *node : sub_graph->Nodes()) {
    if (node->type() == _ArgumentOpType) {
      argument_nodes.push_back(node);
    }
  }
  // Start to divide nodes
  int argument_id = 0;
  for (XrtNode *node : argument_nodes) {
    std::list<XrtEdge *> in_edges = node->in_edges();
    std::list<XrtEdge *> out_edges = node->out_edges();
    // Argument node should has either inputs or outputs
    CHECK(in_edges.size() == 0 || out_edges.size() == 0);

    node->ClearInEdges();
    node->ClearOutEdges();

    util::Map<Argument, XrtNode *> divided_args;
    for (XrtEdge *edge : in_edges) {
      const Argument &arg = edge->argument();
      if (node->in_edges().size() == 0) {
        node->set_name(absl::StrCat(_XrtOutArgumentPrefix, argument_id++));
        divided_args.emplace(arg, node);
      }
      const auto &it = divided_args.find(arg);
      if (it == divided_args.end()) {
        XrtNode *argument = sub_graph->AddNode();
        argument->set_type(_ArgumentOpType);
        argument->set_name(absl::StrCat(_XrtOutArgumentPrefix, argument_id++));
        argument->set_backend(node->backend());
        argument->AddInEdge(edge);
        edge->SetEndNode(argument);
        divided_args.emplace(arg, argument);
      } else {
        XrtNode *consumer = it->second;
        edge->SetEndNode(consumer);
        consumer->AddInEdge(edge);
      }
    }

    for (XrtEdge *edge : out_edges) {
      const Argument &arg = edge->argument();
      if (node->out_edges().size() == 0) {
        node->set_name(absl::StrCat(_XrtInArgumentPrefix, argument_id++));
        divided_args.emplace(arg, node);
      }
      const auto &it = divided_args.find(arg);
      if (it == divided_args.end()) {
        XrtNode *argument = sub_graph->AddNode();
        argument->set_type(_ArgumentOpType);
        argument->set_name(absl::StrCat(_XrtInArgumentPrefix, argument_id++));
        argument->set_backend(node->backend());
        argument->AddOutEdge(edge);
        edge->SetStartNode(argument);
        divided_args.emplace(arg, argument);
      } else {
        XrtNode *producer = it->second;
        edge->SetStartNode(producer);
        producer->AddOutEdge(edge);
      }
    }
  }
}

void BuildSubGraphPass::RebuildSubgraphInputs(
    XrtNode *node, XrtNode *n, XrtGraph *sub_graph,
    util::Map<int64_t, XrtNode *> *sub_graph_nodes) {
  for (XrtEdge *e : n->in_edges()) {
    int64_t start_id = e->start()->unique_id();
    // Check if the edge had been redirected
    if (e->end()->unique_id() != n->unique_id()) {
      XrtNode *argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddNode();
        argument->set_type(_ArgumentOpType);
        argument->set_backend(e->start()->backend());
        sub_graph_nodes->emplace(start_id, argument);
      } else {
        argument = (*sub_graph_nodes)[start_id];
      }
      sub_graph->Connect(argument, node, e->argument());
    } else {
      if (sub_graph_nodes->count(start_id) != 0) {
        XrtNode *start = (*sub_graph_nodes)[start_id];
        sub_graph->Connect(start, node, e->argument());
      }
    }
  }
}

void BuildSubGraphPass::RebuildSubgraphOutputs(
    XrtNode *node, XrtNode *n, XrtGraph *sub_graph,
    util::Map<int64_t, XrtNode *> *sub_graph_nodes) {
  for (XrtEdge *e : n->out_edges()) {
    // Check if the edge had been redirected
    if (e->start()->unique_id() != n->unique_id()) {
      // start_id is the launch node id
      int64_t start_id = e->start()->unique_id();
      XrtNode *argument = nullptr;
      if (sub_graph_nodes->count(start_id) == 0) {
        argument = sub_graph->AddNode();
        argument->set_type(_ArgumentOpType);
        argument->set_backend(e->start()->backend());
        sub_graph_nodes->emplace(start_id, argument);
      } else {
        argument = (*sub_graph_nodes)[start_id];
      }
      sub_graph->Connect(node, argument, e->argument());
    } else {
      int64_t end_id = e->end()->unique_id();
      if (sub_graph_nodes->count(end_id) != 0) {
        XrtNode *end = (*sub_graph_nodes)[end_id];
        sub_graph->Connect(node, end, e->argument());
      }
    }
  }
}

void BuildSubGraphPass::DumpSubgraphs(const XrtGraph *graph,
                                      const std::string &path) {
  for (const XrtNode *node : graph->Nodes()) {
    if (node->type() == _XrtLaunchOpType) {
      std::string file = absl::StrCat(path, "/cluster_", ClusterId(node));
      std::ofstream ost(file.c_str());
      if (ost.good()) ost << node->sub_graph()->ToDot();
    }
  }
}

REGISTER_XRT_PASS(BuildSubGraph, BuildSubGraphPass);

}  // namespace xrt
}  // namespace oneflow
