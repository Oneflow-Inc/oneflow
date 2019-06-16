#include <unordered_set>
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_node.h"

namespace oneflow {
namespace mola {

XlaNode::XlaNode(const OpNode *op_node) : node_(op_node), unique_id_(-1) {
  backend_ = [&]() -> std::string {
      const DeviceType device_type = op_node->op().device_type();
      switch (device_type) {
        case DeviceType::kCPU:
          return "CPU";
        case DeviceType::kGPU:
          return "CUDA";
        default:
          LOG(ERROR) << "Not supported DeviceType (" << device_type
                     << ") in XlaGraphCompiler::Compile";
          return NoneString;
      }
    }();

  op_type_ = ExtractOpTypeAsString(op_node->op());
  compiled_ = IsOpTypeCompiled(backend_, op_type_);

  folded_nodes_.insert(this);
}

void XlaNode::AddInEdge(const XlaEdge *edge) {
  in_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::AddOutEdge(const XlaEdge *edge) {
  out_edges_.push_back(const_cast<XlaEdge *>(edge));
}

std::unordered_set<const XlaNode *> &XlaNode::fold(const XlaNode *other) {
  const auto &other_folds = other->folded_nodes();
  folded_nodes_.insert(other_folds.begin(), other_folds.end());
  return folded_nodes_;
}

bool XlaNode::IsSourceNode() const { return false; }

bool XlaNode::IsFinishNode() const { return false; }

}  // namespace mola
}  // namespace oneflow
