#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

XlaGraph::XlaGraph(const OpGraph *op_graph) {
  CHECK_NOTNULL(op_graph); 
  op_graph->TopoForEachNode([&](OpNode *op_node) -> void {
    AddNode(op_node);
  });

  BuildEdges();
}

void XlaGraph::BuildEdges() {
  std::unordered_map<LogicalBlobId, XlaNode *> lbi2producer;

  for (XlaNode *node : Nodes()) {
    for (const std::string &bn  : node->output_bns()) {
      lbi2producer.emplace(node->Output(bn), node);
    }
  }

  for (XlaNode *node : Nodes()) {
    for (const std::string &bn : node->input_bns()) {
      const LogicalBlobId &lbi = node->Input(bn);
      const auto &it = lbi2producer.find(lbi);
      if (it != lbi2producer.end()) {
        Argument argument(lbi, node->node()->LogicalBlobDesc4Lbi(lbi));
        Connect(it->second, node, argument);
      }
    }
  }
}

XlaEdge *XlaGraph::Connect(XlaNode *start, XlaNode *end) {
  XlaEdge *edge = AddEdge(start, end);
  start->AddOutEdge(edge);
  end->AddInEdge(edge);
  return edge;
}

XlaEdge *XlaGraph::Connect(XlaNode *start, XlaNode *end, const Argument &arg) {
  XlaEdge *edge = Connect(start, end);
  edge->UpdateArgument(arg);
  return edge;
}

void XlaGraph::Disconnect(XlaEdge *edge) {
  edge->start()->EraseOutEdge(edge);
  edge->end()->EraseInEdge(edge);
}

XlaGraph::~XlaGraph() {
  for (auto &node : nodes_) {
    if (NOTNULL(node)) DELETE(node);
  }
  for (auto &edge : edges_) {
    if (NOTNULL(edge)) DELETE(edge);
  }
  for (auto &pair : subgraphs_) {
    DELETE(pair.second);
  }
}

XlaNode *XlaGraph::Node(int64_t node_id) {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

const XlaNode *XlaGraph::Node(int64_t node_id) const {
  DCHECK_LT(node_id, nodes_.size());
  return nodes_[node_id];
}

XlaNode *XlaGraph::AddNode() {
  XlaNode *node = new XlaNode();
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
}

XlaNode *XlaGraph::AddNode(const OpNode *op_node) {
  XlaNode *node = new XlaNode(op_node);
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
}

XlaNode *XlaGraph::AddArgumentNode(const XlaLaunchOpConf::Argument &arg_conf) {
  XlaNode *node = new XlaArgumentNode(arg_conf);
  node->unique_id_ = nodes_.size();
  nodes_.push_back(node);
  return node;
}

XlaEdge *XlaGraph::AddEdge(XlaNode *start, XlaNode *end) {
  XlaEdge *edge = new XlaEdge(start, end);
  edge->unique_id_ = edges_.size();
  edges_.push_back(edge);
  return edge;
}

XlaGraph *XlaGraph::AddSubGraph(int64_t node_id) {
  CHECK_LT(node_id, nodes_.size());
  auto it = subgraphs_.find(node_id);
  if (it != subgraphs_.end()) {
    DELETE(it->second);
    it->second = new XlaGraph;
  } else {
    it = subgraphs_.emplace(node_id, new XlaGraph).first;
  }
  nodes_[node_id]->sub_graph_ = it->second;
  return it->second;
}

void XlaLaunchGraph::SetupArguments() {
  // Add argument nodes
  for (const auto &arg_conf : launch_conf_.attr().argument()) {
    XlaNode *node = this->AddArgumentNode(arg_conf);
    if (node->IsInArgumentNode()) {
      std::string in = BlobName(node->Input("in"));
      inputs_[in] = node->Output("out");
    } else {
      std::string out = BlobName(node->Output("out"));
      outputs_[out] = node->Input("in");
    }
  }
}

static ParallelDesc DefaultParallelDesc() {
  ParallelConf conf;
  return ParallelDesc(conf);
}

void XlaLaunchGraph::BuildLaunchGraph() {
  std::unordered_map<LogicalBlobId, XlaNode *> lbi2producer;
  for (XlaNode *node : this->Nodes()) {
    for (const std::string &bn : node->output_bns()) {
      lbi2producer.emplace(node->Output(bn), node);
    }
  }
  // Add normal nodes
  ParallelDesc parallel_desc = DefaultParallelDesc();
  for (const auto &node_conf : launch_conf_.attr().node()) {
    auto op_node = std::make_shared<OpNode>(parallel_desc, node_conf);
    allocated_opnodes_.push_back(op_node);
    XlaNode *node = this->AddNode(op_node.get());
    for (const std::string &bn : node->output_bns()) {
      lbi2producer.emplace(node->Output(bn), node);
    }
  }
  // Add edges
  for (const auto &node : this->Nodes()) {
    for (const std::string &bn : node->input_bns()) {
      const LogicalBlobId &lbi = node->Input(bn);
      auto it = lbi2producer.find(lbi);
      // Input argument node input lbi maybe equal to output lbi, so here we
      // add edge only if `it->second != node` in order to avoid node-self ring
      if (it != lbi2producer.end() && it->second != node) {
        Argument argument(lbi, BlobDesc());
        this->Connect(it->second, node, argument);
      }
    }
  }
}

void XlaLaunchGraph::InferBlobDescs(
      std::unordered_map<std::string, BlobDesc> *blob_descs,
      const ParallelContext* parallel_ctx) {
  TopologyVisit(*this, [&](XlaNode *node) {
    auto get_blob_desc_fn = [&](const LogicalBlobId &lbi) -> BlobDesc* {
      std::string blob_name = BlobName(lbi);
      auto it = blob_descs->find(blob_name);
      // Check presentness for inputs
      if (IsNodeInput(node, lbi)) {
        CHECK(it != blob_descs->end());
      } else {
        if (it == blob_descs->end()) {
          it = blob_descs->emplace(blob_name, BlobDesc()).first;
        }
      }
      return &(it->second);
    };

    node->InferBlobDescs(get_blob_desc_fn, parallel_ctx);
    // Update blob desc on the output edges
    for (XlaEdge *edge : node->out_edges()) {
      std::string blob_name = edge->argument().blob_name();
      auto it = blob_descs->find(blob_name);
      CHECK(it != blob_descs->end());
      Argument argument(edge->argument().blob_id(), it->second);
      edge->UpdateArgument(argument);
    }
  });
}

}  // namespace mola
}  // namespace oneflow
