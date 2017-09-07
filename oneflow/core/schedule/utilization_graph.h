#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_

#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/utilization.h"

namespace oneflow {
namespace schedule {

class UtilizationGraph final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationGraph);
  explicit UtilizationGraph(const SGraph* sgraph) : sgraph_(sgraph) {}
  ~UtilizationGraph() = default;

  //	getter
  inline const SGraph* sgraph() const { return sgraph_; }
  inline const ComputationUtilization& computation() const {
    return computation_;
  }
  inline const NodeMgr<DeviceComputationUtilization>& dev_computation_mgr()
      const {
    return dev_computation_mgr_;
  }
  inline const NodeMgr<StreamUtilization>& stream_mgr() const {
    return stream_mgr_;
  }
  inline const NodeMgr<TaskUtilization>& task_mgr() const { return task_mgr_; }
  inline const NodeMgr<TaskStreamUtilization>& task_stream_mgr() const {
    return task_stream_mgr_;
  }
  inline const MemoryUtilization& memory() const { return memory_; }
  inline const NodeMgr<DeviceMemoryUtilization>& dev_memory_mgr() const {
    return dev_memory_mgr_;
  }
  inline const NodeMgr<RegstDescUtilization>& regst_desc_mgr() const {
    return regst_desc_mgr_;
  }
  inline const NodeMgr<RegstUtilization>& regst_mgr() const {
    return regst_mgr_;
  }
  inline const ArcMgr<
      Arc<ComputationUtilization, DeviceComputationUtilization>>&
  c2dc_arc_mgr() {
    return c2dc_arc_mgr_;
  }
  inline const ArcMgr<Arc<DeviceComputationUtilization, StreamUtilization>>&
  dc2s_arc_mgr() {
    return dc2s_arc_mgr_;
  }
  inline const ArcMgr<Arc<DeviceComputationUtilization, TaskUtilization>>&
  dc2t_arc_mgr() {
    return dc2t_arc_mgr_;
  }
  inline const ArcMgr<Arc<StreamUtilization, TaskStreamUtilization>>&
  s2ts_arc_mgr() {
    return s2ts_arc_mgr_;
  }
  inline const ArcMgr<Arc<TaskUtilization, TaskStreamUtilization>>&
  t2ts_arc_mgr() {
    return t2ts_arc_mgr_;
  }
  inline const ArcMgr<Arc<MemoryUtilization, DeviceMemoryUtilization>>&
  m2dm_arc_mgr() {
    return m2dm_arc_mgr_;
  }
  inline const ArcMgr<Arc<DeviceMemoryUtilization, RegstDescUtilization>>&
  dm2rd_arc_mgr() {
    return dm2rd_arc_mgr_;
  }
  inline const ArcMgr<Arc<RegstDescUtilization, RegstUtilization>>&
  rd2r_arc_mgr() {
    return rd2r_arc_mgr_;
  }

  //	setter
  inline NodeMgr<DeviceComputationUtilization>* mut_dev_computation_mgr() {
    return &dev_computation_mgr_;
  }
  inline NodeMgr<StreamUtilization>* mut_stream_mgr() { return &stream_mgr_; }
  inline NodeMgr<TaskUtilization>* mut_task_mgr() { return &task_mgr_; }
  inline NodeMgr<TaskStreamUtilization>* mut_task_stream_mgr() {
    return &task_stream_mgr_;
  }
  inline NodeMgr<DeviceMemoryUtilization>* mut_dev_memory_mgr() {
    return &dev_memory_mgr_;
  }
  inline NodeMgr<RegstDescUtilization>* mut_regst_desc_mgr() {
    return &regst_desc_mgr_;
  }
  inline NodeMgr<RegstUtilization>* mut_regst_mgr() { return &regst_mgr_; }
  inline ArcMgr<Arc<ComputationUtilization, DeviceComputationUtilization>>*
  mut_c2dc_arc_mgr() {
    return &c2dc_arc_mgr_;
  }
  inline ArcMgr<Arc<DeviceComputationUtilization, StreamUtilization>>*
  mut_dc2s_arc_mgr() {
    return &dc2s_arc_mgr_;
  }
  inline ArcMgr<Arc<DeviceComputationUtilization, TaskUtilization>>*
  mut_dc2t_arc_mgr() {
    return &dc2t_arc_mgr_;
  }
  inline ArcMgr<Arc<StreamUtilization, TaskStreamUtilization>>*
  mut_s2ts_arc_mgr() {
    return &s2ts_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskUtilization, TaskStreamUtilization>>*
  mut_t2ts_arc_mgr() {
    return &t2ts_arc_mgr_;
  }
  inline ArcMgr<Arc<MemoryUtilization, DeviceMemoryUtilization>>*
  mut_m2dm_arc_mgr() {
    return &m2dm_arc_mgr_;
  }
  inline ArcMgr<Arc<DeviceMemoryUtilization, RegstDescUtilization>>*
  mut_dm2rd_arc_mgr() {
    return &dm2rd_arc_mgr_;
  }
  inline ArcMgr<Arc<RegstDescUtilization, RegstUtilization>>*
  mut_rd2r_arc_mgr() {
    return &rd2r_arc_mgr_;
  }

 private:
  const SGraph* sgraph_;
  ComputationUtilization computation_;
  MemoryUtilization memory_;
  NodeMgr<DeviceComputationUtilization> dev_computation_mgr_;
  NodeMgr<StreamUtilization> stream_mgr_;
  NodeMgr<TaskUtilization> task_mgr_;
  NodeMgr<TaskStreamUtilization> task_stream_mgr_;
  NodeMgr<DeviceMemoryUtilization> dev_memory_mgr_;
  NodeMgr<RegstDescUtilization> regst_desc_mgr_;
  NodeMgr<RegstUtilization> regst_mgr_;
  ArcMgr<Arc<ComputationUtilization, DeviceComputationUtilization>>
      c2dc_arc_mgr_;
  ArcMgr<Arc<DeviceComputationUtilization, StreamUtilization>> dc2s_arc_mgr_;
  ArcMgr<Arc<DeviceComputationUtilization, TaskUtilization>> dc2t_arc_mgr_;
  ArcMgr<Arc<StreamUtilization, TaskStreamUtilization>> s2ts_arc_mgr_;
  ArcMgr<Arc<TaskUtilization, TaskStreamUtilization>> t2ts_arc_mgr_;
  ArcMgr<Arc<MemoryUtilization, DeviceMemoryUtilization>> m2dm_arc_mgr_;
  ArcMgr<Arc<DeviceMemoryUtilization, RegstDescUtilization>> dm2rd_arc_mgr_;
  ArcMgr<Arc<RegstDescUtilization, RegstUtilization>> rd2r_arc_mgr_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_GRAPH_H_
