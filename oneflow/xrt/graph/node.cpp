#include "absl/strings/str_split.h"

#include "oneflow/xrt/graph/algorithm.h"
#include "oneflow/xrt/graph/node.h"

namespace oneflow {
namespace xrt {

void XrtNode::AddInEdge(const XrtEdge *edge) { in_edges_.push_back(const_cast<XrtEdge *>(edge)); }

void XrtNode::AddOutEdge(const XrtEdge *edge) { out_edges_.push_back(const_cast<XrtEdge *>(edge)); }

void XrtNode::EraseInEdge(const XrtEdge *edge) {
  in_edges_.remove_if(
      [&](const XrtEdge *e) -> bool { return e->unique_id() == edge->unique_id(); });
}

void XrtNode::EraseOutEdge(const XrtEdge *edge) {
  out_edges_.remove_if(
      [&](const XrtEdge *e) -> bool { return e->unique_id() == edge->unique_id(); });
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

}  // namespace xrt
}  // namespace oneflow
