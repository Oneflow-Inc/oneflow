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
#ifndef ONEFLOW_CORE_JOB_PORTABLE_CTRL_EDGE_H_
#define ONEFLOW_CORE_JOB_PORTABLE_CTRL_EDGE_H_

#include "oneflow/core/job/portable_ctrl_edge.pb.h"
#include "oneflow/core/common/hash.h"

namespace oneflow {

inline bool operator==(const PortableCtrlNode& lhs, const PortableCtrlNode& rhs) {
  if (lhs.ctrl_node_type_case() != rhs.ctrl_node_type_case()) { return false; }
  if (lhs.ctrl_node_type_case() == PortableCtrlNode::CTRL_NODE_TYPE_NOT_SET) { return true; }
  if (lhs.has_transport_task_id()) {
    return lhs.transport_task_id() == rhs.transport_task_id();
  } else if (lhs.has_compute_task_op_name()) {
    return lhs.compute_task_op_name() == rhs.compute_task_op_name();
  } else {
    UNIMPLEMENTED();
  }
  return false;
}

inline bool operator!=(const PortableCtrlNode& lhs, const PortableCtrlNode& rhs) {
  return !(lhs == rhs);
}

inline bool operator==(const PortableCtrlEdge& lhs, const PortableCtrlEdge& rhs) {
  return lhs.src() == rhs.src() && lhs.dst() == rhs.dst();
}

inline bool operator!=(const PortableCtrlEdge& lhs, const PortableCtrlEdge& rhs) {
  return !(lhs == rhs);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::PortableCtrlNode> final {
  size_t operator()(const oneflow::PortableCtrlNode& node) const {
    using namespace oneflow;
    if (node.ctrl_node_type_case() == PortableCtrlNode::CTRL_NODE_TYPE_NOT_SET) { return 0; }
    if (node.has_transport_task_id()) {
      return std::hash<int64_t>()(node.transport_task_id());
    } else if (node.has_compute_task_op_name()) {
      return std::hash<std::string>()(node.compute_task_op_name());
    } else {
      UNIMPLEMENTED();
    }
  }
};

template<>
struct hash<oneflow::PortableCtrlEdge> final {
  size_t operator()(const oneflow::PortableCtrlEdge& edge) const {
    return oneflow::Hash(edge.src(), edge.dst());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_PORTABLE_CTRL_EDGE_H_
