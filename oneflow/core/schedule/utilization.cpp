#include "oneflow/core/schedule/utilization.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

void Utilization::Reduce(const UtilizationGraph& graph) {
  uint32_t parallel_num = ParallelNum(graph);
  CHECK(parallel_num);
  float total_time = 0, start_at = UINT64_MAX, end_at = 0;
  uint64_t start_batch_id = UINT64_MAX, end_batch_id = 0;
  for (const UtilizationProto* u : raw_protos_) {
    total_time += u->end_at() - u->start_at();
    start_at = std::min(start_at, u->start_at());
    end_at = std::max(end_at, u->end_at());
    start_batch_id = std::min(start_batch_id, u->start_batch_id());
    end_batch_id = std::max(end_batch_id, u->end_batch_id());
  }
  CHECK(end_at > start_at);
  float utilization = total_time / parallel_num * (end_at - start_at);
  CHECK(total_time < 1);
  utilization_proto_.set_utilization(utilization);
  utilization_proto_.set_start_at(start_at);
  utilization_proto_.set_start_at(start_at);
  utilization_proto_.set_end_at(end_at);
  utilization_proto_.set_start_batch_id(start_batch_id);
  utilization_proto_.set_end_batch_id(end_batch_id);
}

uint32_t MemoryUtilization::ParallelNum(const UtilizationGraph& graph) const {
  return graph.m2leaf_arc_mgr().Output(const_cast<MemoryUtilization*>(this));
}

uint32_t DeviceComputationUtilization::ParallelNum(
    const UtilizationGraph& graph) const {
  return graph.dc2s_arc_mgr().Output(
      const_cast<DeviceComputationUtilization*>(this));
}

void DeviceComputationUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph, TaskStreamUtilization* leaf) const {
  graph->mut_root2dc_arc_mgr()->CreateIfNotFound(
      const_cast<ComputationUtilization*>(&graph->computation()),
      const_cast<DeviceComputationUtilization*>(this));
  graph->mut_c2leaf_arc_mgr()->CreateIfNotFound(
      const_cast<ComputationUtilization*>(&graph->computation()), leaf);
}

void StreamUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph, TaskStreamUtilization* leaf) const {
  auto name = DeviceComputationUtilization::MakeUniqueName(device_id());
  if (graph->dev_computation_mgr().Find(name)) return;
  auto node = graph->mut_dev_computation_mgr()->Create(device_id());
  graph->mut_dc2s_arc_mgr()->CreateIfNotFound(
      node, const_cast<StreamUtilization*>(this));
  graph->mut_c2leaf_arc_mgr()->CreateIfNotFound(node, leaf);
  node->CreateAscendantIfNotFound(graph, leaf);
}

void TaskUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph, TaskStreamUtilization* leaf) const {
  auto task = graph->sgraph()->node_mgr().Find(task_id());
  uint64_t device_id = task->device()->id();
  auto name = DeviceComputationUtilization::MakeUniqueName(device_id);
  if (graph->dev_computation_mgr().Find(name)) return;
  auto node = graph->mut_dev_computation_mgr()->Create(device_id);
  graph->mut_dc2t_arc_mgr()->CreateIfNotFound(
      node, const_cast<TaskUtilization*>(this));
  graph->mut_c2leaf_arc_mgr()->CreateIfNotFound(node, leaf);
  node->CreateAscendantIfNotFound(graph, leaf);
}

void TaskStreamUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph, TaskStreamUtilization* leaf) const {
  auto task_name = TaskUtilization::MakeUniqueName(task_id());
  if (graph->task_mgr().Find(task_name)) return;
  auto task_node = graph->mut_task_mgr()->Create(task_id());
  graph->mut_t2ts_arc_mgr()->CreateIfNotFound(
      task_node, const_cast<TaskStreamUtilization*>(this));
  graph->mut_c2leaf_arc_mgr()->CreateIfNotFound(task_node, leaf);
  task_node->CreateAscendantIfNotFound(graph, leaf);

  auto task = graph->sgraph()->node_mgr().Find(task_id());
  uint64_t device_id = task->device()->id();
  auto stream_name = StreamUtilization::MakeUniqueName(device_id, stream_id());
  if (graph->stream_mgr().Find(stream_name)) return;
  auto stream_node = graph->mut_stream_mgr()->Create(device_id, stream_id());
  graph->mut_s2ts_arc_mgr()->CreateIfNotFound(
      stream_node, const_cast<TaskStreamUtilization*>(this));
  graph->mut_c2leaf_arc_mgr()->CreateIfNotFound(stream_node, leaf);
  stream_node->CreateAscendantIfNotFound(graph, leaf);
}

void DeviceMemoryUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph, RegstUtilization* leaf) const {
  graph->mut_root2dm_arc_mgr()->CreateIfNotFound(
      const_cast<MemoryUtilization*>(&graph->memory()),
      const_cast<DeviceMemoryUtilization*>(this));
  graph->mut_m2leaf_arc_mgr()->CreateIfNotFound(
      const_cast<MemoryUtilization*>(&graph->memory()), leaf);
}

void RegstDescUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph, RegstUtilization* leaf) const {
  auto regst_desc = graph->sgraph()->regst_desc_mgr().Find(regst_desc_id());
  uint64_t device_id = regst_desc->owner_task()->device()->id();
  auto name = DeviceMemoryUtilization::MakeUniqueName(device_id);
  if (graph->dev_memory_mgr().Find(name)) return;
  auto node = graph->mut_dev_memory_mgr()->Create(device_id);
  graph->mut_dm2rd_arc_mgr()->CreateIfNotFound(
      node, const_cast<RegstDescUtilization*>(this));
  graph->mut_m2leaf_arc_mgr()->CreateIfNotFound(node, leaf);
  node->CreateAscendantIfNotFound(graph, leaf);
}

void RegstUtilization::CreateAscendantIfNotFound(UtilizationGraph* graph,
                                                 RegstUtilization* leaf) const {
  leaf = const_cast<RegstUtilization*>(this);
  auto name = RegstDescUtilization::MakeUniqueName(regst_desc_id());
  if (graph->regst_desc_mgr().Find(name)) return;
  auto node = graph->mut_regst_desc_mgr()->Create(regst_desc_id());
  graph->mut_rd2r_arc_mgr()->CreateIfNotFound(
      node, const_cast<RegstUtilization*>(this));
  graph->mut_m2leaf_arc_mgr()->CreateIfNotFound(node, leaf);
  node->CreateAscendantIfNotFound(graph, leaf);
}

}  // namespace schedule
}  // namespace oneflow
