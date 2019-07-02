#include <unordered_set>
#include "absl/strings/str_split.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_node.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"

namespace oneflow {
namespace mola {

extern const std::string _XlaArgumentOpType;
extern const std::string _XlaInArgumentPrefix;
extern const std::string _XlaOutArgumentPrefix;

XlaNode::XlaNode(const OpNode *op_node) : node_(op_node), unique_id_(-1),
                                          cluster_id_(-1), sub_graph_(nullptr) {
  backend_ = [&]() -> std::string {
      const DeviceType device_type = op_node->op().device_type();
      switch (device_type) {
        case DeviceType::kCPU:
          return "CPU";
        case DeviceType::kGPU:
          return "CUDA";
        default:
          LOG(ERROR) << "Not supported DeviceType (" << device_type << ")";
          return NoneString;
      }
    }();

  op_name_ = op_node->op().op_name();
  op_type_ = ExtractOpTypeAsString(op_node->op().op_conf());
  compiled_ = IsOpCompilerRegistered(backend_, op_type_);
  // Setup input and output logical blob ids
  for (const std::string &bn : op_node->op().input_bns()) {
    const LogicalBlobId &lbi = op_node->op().BnInOp2Lbi(bn);
    inputs_.push_back(lbi);
  }
  for (const std::string &bn : op_node->op().output_bns()) {
    const LogicalBlobId &lbi = op_node->op().BnInOp2Lbi(bn);
    outputs_.push_back(lbi);
  }
}

void XlaNode::AddInEdge(const XlaEdge *edge) {
  in_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::AddOutEdge(const XlaEdge *edge) {
  out_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::EraseInEdge(const XlaEdge *edge) {
  for (auto it = in_edges_.begin(); it != in_edges_.end(); ++it) {
    if ((*it)->unique_id() == edge->unique_id() &&
        (*it)->argument() == edge->argument()) {
      in_edges_.erase(it);
    }
  }
}

void XlaNode::EraseOutEdge(const XlaEdge *edge) {
  for (auto it = out_edges_.begin(); it != out_edges_.end(); ++it) {
    if ((*it)->unique_id() == edge->unique_id() &&
        (*it)->argument() == edge->argument()) {
      out_edges_.erase(it);
    }
  }
}

void XlaNode::InferBlobDescs(GetBlobDescFunc func,
                             const ParallelContext* parallel_ctx) const {
  auto inner_get_blob_desc_fn = [&](const std::string &bn) -> BlobDesc* {
    const LogicalBlobId &lbi = op()->BnInOp2Lbi(bn);
    return func(lbi);
  };
  op()->InferBlobDescs(inner_get_blob_desc_fn, parallel_ctx);
}

bool XlaNode::IsSourceNode() const {
  // TODO(hjchen2)
  return false;
}

bool XlaNode::IsFinishNode() const {
  // TODO(hjchen2)
  return false;
}

bool XlaNode::IsArgumentNode() const {
  return op_type_ == _XlaArgumentOpType;
}

XlaArgumentNode::XlaArgumentNode(const XlaLaunchOpConf::Argument &arg_conf)
    : XlaNode() {
  this->op_type_ = _XlaArgumentOpType;
  this->op_name_ = arg_conf.name();
  this->compiled_ = true;
  if (absl::StartsWith(arg_conf.name(), _XlaInArgumentPrefix)) {
    for (const std::string &bn : arg_conf.out()) {
      LogicalBlobId lbi = GenLogicalBlobId(bn);
      this->outputs_.push_back(lbi);
    }
  } else {
     for (const std::string &bn : arg_conf.in()) {
      LogicalBlobId lbi = GenLogicalBlobId(bn);
      this->inputs_.push_back(lbi);
    }
  }
}

void XlaArgumentNode::InferBlobDescs(
    GetBlobDescFunc func, const ParallelContext* parallel_ctx) const {
  for (const LogicalBlobId &input : this->inputs_) {
    func(input);
  }
}

bool IsNodeInput(const XlaNode *node, const LogicalBlobId &lbi) {
  const auto &inputs = node->Input();
  return (std::find(inputs.begin(), inputs.end(), lbi) != inputs.end());
}

bool IsNodeOutput(const XlaNode *node, const LogicalBlobId &lbi) {
  const auto &outputs = node->Output();
  return (std::find(outputs.begin(), outputs.end(), lbi) != outputs.end());
}

}  // namespace mola
}  // namespace oneflow
