#include <unordered_set>
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_node.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"

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
  compiled_ = XlaOpCompilerRegistry::IsRegistered(backend_, op_type_);
}

void XlaNode::AddInEdge(const XlaEdge *edge) {
  in_edges_.push_back(const_cast<XlaEdge *>(edge));
}

void XlaNode::AddOutEdge(const XlaEdge *edge) {
  out_edges_.push_back(const_cast<XlaEdge *>(edge));
}

bool XlaNode::IsSourceNode() const { return false; }

bool XlaNode::IsFinishNode() const { return false; }

}  // namespace mola
}  // namespace oneflow
