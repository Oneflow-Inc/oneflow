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
#ifndef ONEFLOW_CORE_GRAPH_TRANSPORT_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_TRANSPORT_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

class TransportTaskProto;
class TaskGraphRebuildCtx;

class TransportTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransportTaskNode);
  TransportTaskNode() = default;
  virtual ~TransportTaskNode() = default;

  void set_lbi(const LogicalBlobId& lbi) { lbi_ = lbi; }
  LogicalBlobId lbi() const { return lbi_; }

  Maybe<void> InitTransportTaskFromProtoIf(const TransportTaskProto& transport_task_proto,
                                           const TaskGraphRebuildCtx& ctx);
  void ToTransportTaskProtoIf(TransportTaskProto*) const;

  ExecNode::InferBlobDescsMethod GetInferBlobDescsMethod() const override {
    // TransportTaskNode infers output BlobDesc based on input BlobDesc, because it can't infers
    // output BlobDesc with SBP.
    return &ExecNode::InferBlobDescsByInputs;
  }

 private:
  virtual Maybe<void> InitTransportTaskFromProto(const TransportTaskProto&,
                                                 const TaskGraphRebuildCtx& ctx) = 0;

  virtual void ToTransportTaskProto(TransportTaskProto*) const = 0;
  LogicalBlobId lbi_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TRANSPORT_TASK_NODE_H_
