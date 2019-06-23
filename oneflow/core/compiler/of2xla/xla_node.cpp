#include <unordered_set>
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_node.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"

namespace oneflow {
namespace mola {

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
}

void XlaNode::AddInEdge(const XlaEdge *edge) {
  in_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::AddOutEdge(const XlaEdge *edge) {
  out_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::EraseInEdge(const XlaEdge *edge) {
  for (auto it = in_edges_.begin(); it != in_edges_.end(); ++it) {
    if ((*it)->unique_id() == edge->unique_id()) {
      in_edges_.erase(it);
    }
  }
}

void XlaNode::EraseOutEdge(const XlaEdge *edge) {
  for (auto it = out_edges_.begin(); it != out_edges_.end(); ++it) {
    if ((*it)->unique_id() == edge->unique_id()) {
      out_edges_.erase(it);
    }
  }
}

bool XlaNode::IsSourceNode() const {
  // TODO(hjchen2)
  return false;
}

bool XlaNode::IsFinishNode() const {
  // TODO(hjchen2)
  return false;
}

bool IsControlEdge(const XlaEdge *edge) {
  // TODO(hjchen2)
  return false;
}

}  // namespace mola
}  // namespace oneflow
