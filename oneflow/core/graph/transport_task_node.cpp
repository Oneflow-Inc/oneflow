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
#include "oneflow/core/graph/transport_task_node.h"
#include "oneflow/core/graph/boxing_task_graph.pb.h"

namespace oneflow {

Maybe<void> TransportTaskNode::InitTransportTaskFromProtoIf(
    const TransportTaskProto& transport_task_proto, const TaskGraphRebuildCtx& ctx) {
  CHECK(has_new_task_id());
  JUST(InitTransportTaskFromProto(transport_task_proto, ctx));
  lbi_ = transport_task_proto.lbi();
  return Maybe<void>::Ok();
}

void TransportTaskNode::ToTransportTaskProtoIf(TransportTaskProto* transport_task_proto) const {
  ToTransportTaskProto(transport_task_proto);
  *transport_task_proto->mutable_lbi() = lbi_;
}

}  // namespace oneflow
