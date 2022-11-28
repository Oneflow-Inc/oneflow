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
#ifndef ONEFLOW_CORE_GRAPH_NCCL_SEND_RECV_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_SEND_RECV_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/transport_task_node.h"

namespace oneflow {

class NcclSendRecvBoxingTaskNode : public TransportTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclSendRecvBoxingTaskNode);
  NcclSendRecvBoxingTaskNode() = default;
  ~NcclSendRecvBoxingTaskNode() override = default;

  void Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
            const Shape& logical_shape, const DataType& data_type, const NdSbp& src_nd_sbp,
            const NdSbp& dst_nd_sbp, const ParallelDesc& src_parallel_desc,
            const ParallelDesc& dst_parallel_desc, const int64_t parallel_id,
            const ParallelDesc& parallel_desc, const bool has_input, const bool has_output,
            const std::string& stream_name);
  TaskType GetTaskType() const override { return TaskType::kNcclSendRecvBoxing; }
  const ParallelContext* parallel_ctx() const override { return &parallel_ctx_; }

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;

  Shape logical_shape_;
  DataType data_type_;
  NdSbp src_nd_sbp_;
  NdSbp dst_nd_sbp_;
  ParallelConf src_parallel_conf_;
  ParallelConf dst_parallel_conf_;
  ParallelConf parallel_conf_;
  ParallelContext parallel_ctx_;
  bool has_input_;
  bool has_output_;
  std::string stream_name_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_SEND_RECV_BOXING_TASK_NODE_H_
