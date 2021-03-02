/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
