#include "oneflow/xrt/graph/node.h"
#include "absl/strings/str_split.h"
#include "oneflow/xrt/graph/algorithm.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

extern const std::string _ArgumentOpType;
extern const std::string _XrtInArgumentPrefix;
extern const std::string _XrtOutArgumentPrefix;

bool XrtNode::IsCompiled(const XrtEngine &engine) const {
  auto field = std::make_pair(engine, backend_);
  auto *rm = util::RegistryManager<decltype(field)>::Global();
  return rm->Get(field)->IsRegistered(type_);
}

void XrtNode::AddInEdge(const XrtEdge *edge) {
  in_edges_.push_back(const_cast<XrtEdge *>(edge));
}

void XrtNode::AddOutEdge(const XrtEdge *edge) {
  out_edges_.push_back(const_cast<XrtEdge *>(edge));
}

void XrtNode::EraseInEdge(const XrtEdge *edge) {
  in_edges_.remove_if([&](const XrtEdge *e) -> bool {
    return e->unique_id() == edge->unique_id();
  });
}

void XrtNode::EraseOutEdge(const XrtEdge *edge) {
  out_edges_.remove_if([&](const XrtEdge *e) -> bool {
    return e->unique_id() == edge->unique_id();
  });
}

bool XrtNode::IsSourceNode() const { return in_edges_.size() == 0; }

bool XrtNode::IsFinishNode() const { return out_edges_.size() == 0; }

bool XrtNode::IsArgumentNode() const { return type_ == _ArgumentOpType; }

bool XrtNode::IsInArgumentNode() const {
  return IsArgumentNode() && absl::StartsWith(name_, _XrtInArgumentPrefix);
}

bool XrtNode::IsOutArgumentNode() const {
  return IsArgumentNode() && absl::StartsWith(name_, _XrtOutArgumentPrefix);
}

bool XrtNode::IsReachable(const XrtNode &dst_node) const {
  return algorithm::IsReachable(this, &dst_node);
}

bool IsNodeInput(const XrtNode *node, const Argument &argument) {
  for (XrtEdge *edge : node->in_edges()) {
    if (edge->argument() == argument) {
      return true;
    }
  }
  return false;
}

bool IsNodeOutput(const XrtNode *node, const Argument &argument) {
  for (XrtEdge *edge : node->out_edges()) {
    if (edge->argument() == argument) {
      return true;
    }
  }
  return false;
}

}  // namespace xrt
}  // namespace oneflow
