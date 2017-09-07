#include "oneflow/core/schedule/utilization.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

void DeviceComputationUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph) const {
  graph->mut_c2dc_arc_mgr()->CreateIfNotFound(
      const_cast<ComputationUtilization*>(&graph->computation()),
      const_cast<DeviceComputationUtilization*>(this));
}

void StreamUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph) const {
  auto name = DeviceComputationUtilization::MakeUniqueName(device_id());
  if (graph->dev_computation_mgr().Find(name)) return;
  auto node = graph->mut_dev_computation_mgr()->Create(device_id());
  graph->mut_dc2s_arc_mgr()->CreateIfNotFound(
      node, const_cast<StreamUtilization*>(this));
  node->CreateAscendantIfNotFound(graph);
}

void TaskUtilization::CreateAscendantIfNotFound(UtilizationGraph* graph) const {
  auto task = graph->sgraph()->node_mgr().Find(task_id());
  uint64_t device_id = task->device()->id();
  auto name = DeviceComputationUtilization::MakeUniqueName(device_id);
  if (graph->dev_computation_mgr().Find(name)) return;
  auto node = graph->mut_dev_computation_mgr()->Create(device_id);
  graph->mut_dc2t_arc_mgr()->CreateIfNotFound(
      node, const_cast<TaskUtilization*>(this));
  node->CreateAscendantIfNotFound(graph);
}

void TaskStreamUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph) const {
  auto task_name = TaskUtilization::MakeUniqueName(task_id());
  if (graph->task_mgr().Find(task_name)) return;
  auto task_node = graph->mut_task_mgr()->Create(task_id());
  graph->mut_t2ts_arc_mgr()->CreateIfNotFound(
      task_node, const_cast<TaskStreamUtilization*>(this));
  task_node->CreateAscendantIfNotFound(graph);

  auto task = graph->sgraph()->node_mgr().Find(task_id());
  uint64_t device_id = task->device()->id();
  auto stream_name = StreamUtilization::MakeUniqueName(device_id, stream_id());
  if (graph->stream_mgr().Find(stream_name)) return;
  auto stream_node = graph->mut_stream_mgr()->Create(device_id, stream_id());
  graph->mut_s2ts_arc_mgr()->CreateIfNotFound(
      stream_node, const_cast<TaskStreamUtilization*>(this));
  stream_node->CreateAscendantIfNotFound(graph);
}

void DeviceMemoryUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph) const {
  graph->mut_m2dm_arc_mgr()->CreateIfNotFound(
      const_cast<MemoryUtilization*>(&graph->memory()),
      const_cast<DeviceMemoryUtilization*>(this));
}

void RegstDescUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph) const {
  auto regst_desc = graph->sgraph()->regst_desc_mgr().Find(regst_desc_id());
  uint64_t device_id = regst_desc->owner_task()->device()->id();
  auto name = DeviceMemoryUtilization::MakeUniqueName(device_id);
  if (graph->dev_memory_mgr().Find(name)) return;
  auto node = graph->mut_dev_memory_mgr()->Create(device_id);
  graph->mut_dm2rd_arc_mgr()->CreateIfNotFound(
      node, const_cast<RegstDescUtilization*>(this));
  node->CreateAscendantIfNotFound(graph);
}

void RegstUtilization::CreateAscendantIfNotFound(
    UtilizationGraph* graph) const {
  auto name = RegstDescUtilization::MakeUniqueName(regst_desc_id());
  if (graph->regst_desc_mgr().Find(name)) return;
  auto node = graph->mut_regst_desc_mgr()->Create(regst_desc_id());
  graph->mut_rd2r_arc_mgr()->CreateIfNotFound(
      node, const_cast<RegstUtilization*>(this));
  node->CreateAscendantIfNotFound(graph);
}

}  // namespace schedule
}  // namespace oneflow
