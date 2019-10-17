#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/xla/of2xla/xla_argument.h"
#include "oneflow/xla/of2xla/xla_utility.h"

namespace oneflow {
namespace mola {

static const BlobDesc &LbiBlobDesc(const OpNode *op_node,
                                   const LogicalBlobId &lbi) {
  CHECK_NOTNULL(op_node);
  return op_node->LogicalBlobDesc4Lbi(lbi);
}

static const Shape &InputTimeShape(const OpNode *op_node) {
  CHECK_NOTNULL(op_node);
  return *(op_node->GetInputBlobFastestTimeShape());
}

static const Shape &OutputTimeShape(const OpNode *op_node) {
  CHECK_NOTNULL(op_node);
  return *(op_node->out_blob_time_shape());
}

static const SbpParallel &LbiSbpPolicy(const OpNode *op_node,
                                       const LogicalBlobId &lbi) {
  CHECK_NOTNULL(op_node);
  return op_node->SbpParallel4Lbi(lbi);
}

XlaGraph::XlaGraph(const OpGraph *op_graph) {
  CHECK_NOTNULL(op_graph);
  std::unordered_map<XlaNode *, const OpNode *> nodes;
  op_graph->TopoForEachNode([&, this](const OpNode *op_node) {
    XlaNode *node = AddNode(&(op_node->op()));
    nodes.emplace(node, op_node);
  });

  std::unordered_map<LogicalBlobId, XlaNode *> producer;
  for (XlaNode *node : Nodes()) {
    for (const std::string &bn : node->output_bns()) {
      producer.emplace(node->Output(bn), node);
    }
  }

  for (XlaNode *node : Nodes()) {
    for (const std::string &bn : node->input_bns()) {
      const LogicalBlobId &lbi = node->Input(bn);
      const auto &it = producer.find(lbi);
      if (it != producer.end()) {
        CHECK(nodes.count(it->second));
        CHECK(nodes.count(node));
        const OpNode *src_node = nodes[it->second];
        const OpNode *dst_node = nodes[node];
        Argument argument(lbi, LbiBlobDesc(dst_node, lbi));
        XlaEdge *edge = Connect(it->second, node, argument);

        CHECK_NOTNULL(edge);
        edge->set_time_shape(0, OutputTimeShape(src_node));
        edge->set_time_shape(1, InputTimeShape(dst_node));
        edge->set_sbp_policy(0, LbiSbpPolicy(src_node, lbi));
        edge->set_sbp_policy(1, LbiSbpPolicy(dst_node, lbi));
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

XlaNode *XlaGraph::AddNode(const Operator *op) {
  XlaNode *node = new XlaNode(op);
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

std::vector<Argument> XlaGraph::Arguments() const {
  std::vector<Argument> arguments;
  for (const XlaEdge *edge : edges_) {
    if (edge) {
      arguments.push_back(edge->argument());
    }
  }
  return std::move(arguments);
}

std::string XlaGraph::ToDot() const {
  std::stringstream ost;
  ost << "digraph {\n";
  for (const XlaNode *node : this->Nodes()) {
    ost << "\"" << node->unique_id() << "\" [label=\"" << node->op_name()
        << "\"]\n";
  }
  for (const XlaEdge *edge : edges_) {
    ost << "\"" << edge->start()->unique_id() << "\" -> "
        << "\"" << edge->end()->unique_id() << "\"\n";
  }
  ost << "}";
  return ost.str();
}

void XlaGraph::InferBlobDescs(
    std::unordered_map<std::string, BlobDesc> *blob_descs,
    const ParallelContext &parallel_ctx, const SbpSignature &sbp_signature) {
  TopologyVisit(*this, [&](XlaNode *node) {
    auto get_blob_desc_fn = [&](const LogicalBlobId &lbi) -> BlobDesc * {
      std::string blob_name = BlobName(lbi);
      auto it = blob_descs->find(blob_name);
      // Check presentness for inputs
      if (IsNodeInput(node, lbi)) {
        CHECK(it != blob_descs->end());
      } else {
        if (it == blob_descs->end()) {
          it = blob_descs->emplace(blob_name, BlobDesc(kFloat)).first;
        }
      }
      return &(it->second);
    };

    node->InferBlobDescs(get_blob_desc_fn, parallel_ctx, sbp_signature);
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

void XlaLaunchGraph::BuildLaunchGraph() {
  std::unordered_map<LogicalBlobId, XlaNode *> lbi2producer;
  for (XlaNode *node : this->Nodes()) {
    for (const std::string &bn : node->output_bns()) {
      lbi2producer.emplace(node->Output(bn), node);
    }
  }
  // Add normal nodes
  for (const auto &node_conf : launch_conf_.attr().node()) {
    std::shared_ptr<Operator> op = ConstructOp(node_conf, job_desc_);
    allocated_ops_.push_back(op);
    XlaNode *node = this->AddNode(op.get());
    for (const std::string &bn : node->output_bns()) {
      lbi2producer.emplace(node->Output(bn), node);
    }
  }

  // const auto &resource_scope = launch_conf_.attr().resource_scope();
  // const auto &shapes = resource_scope.shapes();
  // Add edges
  for (const auto &node : this->Nodes()) {
    for (const std::string &bn : node->input_bns()) {
      const LogicalBlobId &lbi = node->Input(bn);
      const std::string blob_name = BlobName(lbi);
      auto it = lbi2producer.find(lbi);
      // Input argument node input lbi maybe equal to output lbi, so here we
      // add edge only if `it->second != node` to avoid node-self ring
      if (it != lbi2producer.end() && it->second != node) {
        // const auto &shape = shapes.at(blob_name);
        BlobDesc blob_desc(job_desc_->DefaultDataType());
        // blob_desc.set_data_type(shape.data_type());
        // blob_desc.mut_shape() = Shape(shape.shape());
        Argument argument(lbi, blob_desc);
        this->Connect(it->second, node, argument);
      }
    }
  }
}

}  // namespace mola
}  // namespace oneflow
