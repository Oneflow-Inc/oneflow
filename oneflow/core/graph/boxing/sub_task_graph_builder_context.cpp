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
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"

namespace oneflow {

SubTskGphBuilderCtx::SubTskGphBuilderCtx(TaskGraph* task_graph) : task_graph_(task_graph) {}

TaskGraph* SubTskGphBuilderCtx::task_graph() { return task_graph_; }

TaskNode* SubTskGphBuilderCtx::GetProxyNode(TaskNode* src_node, MemZoneId src_mem_zone_id,
                                            int64_t dst_machine_id, MemZoneId dst_mem_zone_id) {
  const auto key = std::make_pair(dst_machine_id, dst_mem_zone_id);
  if (node2proxies_.find(src_node) != node2proxies_.cend()
      && node2proxies_.at(src_node).find(key) != node2proxies_.at(src_node).cend()) {
    return node2proxies_.at(src_node).at(key);
  } else {
    if (dst_machine_id == src_node->machine_id()
        && MemZoneIdUtil::IsMemZoneIdSameDevice(dst_mem_zone_id, src_mem_zone_id)) {
      node2proxies_[src_node][key] = src_node;
      return src_node;
    } else if (MemZoneIdUtil::IsCudaMemZoneId(dst_mem_zone_id)) {
      TaskNode* proxy_on_dst_host =
          GetProxyNode(src_node, src_mem_zone_id, dst_machine_id, MemZoneIdUtil::GetCpuMemZoneId());
      CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, proxy_on_dst_host->machine_id(),
                      MemZoneIdUtil::GetGpuPhyIdFromMemZoneId(dst_mem_zone_id));
      Connect<TaskNode>(proxy_on_dst_host, task_graph()->NewEdge(), copy_task);
      node2proxies_[src_node][key] = copy_task;
      return copy_task;
    } else if (MemZoneIdUtil::IsCpuMemZoneId(dst_mem_zone_id)) {
      if (src_node->machine_id() == dst_machine_id) {
        if (MemZoneIdUtil::IsCudaMemZoneId(src_mem_zone_id)) {
          CopyHdTaskNode* copy_task = task_graph()->NewNode<CopyHdTaskNode>();
          copy_task->Init(CopyHdOpConf::D2H, src_node->machine_id(),
                          MemZoneIdUtil::GetGpuPhyIdFromMemZoneId(src_mem_zone_id));
          Connect<TaskNode>(src_node, task_graph()->NewEdge(), copy_task);
          node2proxies_[src_node][key] = copy_task;
          return copy_task;
        } else {
          UNIMPLEMENTED();
        }
      } else {
        TaskNode* proxy_on_src_host = GetProxyNode(
            src_node, src_mem_zone_id, src_node->machine_id(), MemZoneIdUtil::GetCpuMemZoneId());
        CopyCommNetTaskNode* copy_comm_net_task = task_graph()->NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task->Init(dst_machine_id);
        Connect<TaskNode>(proxy_on_src_host, task_graph()->NewEdge(), copy_comm_net_task);
        node2proxies_[src_node][key] = copy_comm_net_task;
        return copy_comm_net_task;
      }
    } else {
      UNIMPLEMENTED();
    }
  }
}

TaskNode* SubTskGphBuilderCtx::GetProxyNode(TaskNode* src_node, const MemZoneId src_mem_zone_id,
                                            const ParallelDesc& dst_parallel_desc,
                                            const int64_t dst_parallel_id) {
  const int64_t dst_machine_id =
      CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(dst_parallel_id));
  MemZoneId dst_mem_zone_id;
  if (dst_parallel_desc.device_type() == DeviceType::kCPU) {
    dst_mem_zone_id = MemZoneIdUtil::GetCpuMemZoneId();
  } else if (dst_parallel_desc.device_type() == DeviceType::kGPU) {
    const int64_t dst_dev_phy_id =
        CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(dst_parallel_id));
    dst_mem_zone_id = MemZoneIdUtil::GetDeviceMemZoneId(DeviceType::kGPU, dst_dev_phy_id);
  } else {
    UNIMPLEMENTED();
  }
  return GetProxyNode(src_node, src_mem_zone_id, dst_machine_id, dst_mem_zone_id);
}

}  // namespace oneflow
