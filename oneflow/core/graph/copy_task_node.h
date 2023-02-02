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
#ifndef ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_

#include "oneflow/core/graph/transport_task_node.h"

namespace oneflow {

class CopyTaskNode : public TransportTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyTaskNode);
  CopyTaskNode() = default;
  virtual ~CopyTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

 protected:
  virtual OperatorConf NewCopyOpConf() = 0;

 private:
  void InferProducedDataRegstTimeShape() final;
};

enum CopyHdType { H2D = 0, D2H = 1 };

class CopyHdTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdTaskNode);
  CopyHdTaskNode() = default;
  ~CopyHdTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyHd; }

  void Init(CopyHdType, const DeviceId& device_id, const LogicalBlobId& lbi);

  void ProduceAllRegstsAndBindEdges() override;

  CopyHdType copy_type() const { return copy_type_; }
  MemZoneId MemZoneId121() const override {
    if (copy_type_ == CopyHdType::H2D) {
      return TaskNode::MemZoneId121();
    } else if (copy_type_ == CopyHdType::D2H) {
      return GetNodeCPUMemZoneId(this->machine_id());
    } else {
      UNIMPLEMENTED();
    }
    return kInvalidMemZoneId;
  }

 private:
  void InitProducedRegstMemCase(MemoryCase*) override;
  OperatorConf NewCopyOpConf() override;

  CopyHdType copy_type_;
};

class CopyCommNetTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetTaskNode);
  CopyCommNetTaskNode() = default;
  ~CopyCommNetTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyCommNet; }

  void Init(int64_t machine_id, const LogicalBlobId& lbi);

 private:
  OperatorConf NewCopyOpConf() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
