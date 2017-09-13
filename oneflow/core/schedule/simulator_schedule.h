#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"
#include "oneflow/core/schedule/utilization.pb.h"

namespace oneflow {
namespace schedule {

class SimulatorScheduleEngine;

class SimulatorSchedule : public Schedule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SimulatorSchedule);
  explicit SimulatorSchedule(const Session* session) : Schedule(session) {}

  void TimeLinePushBack(TaskInstance* instance, SDevice* device);
  void Retiming();
  void InitTimeNet();
  void EmitBeforeRunEvent(TaskInstance* instance, float time) {
    EmitEvent(UtilizationEventType::kStartEvent, instance, time);
  }
  void EmitAfterRunEvent(TaskInstance* instance, float time) {
    EmitEvent(UtilizationEventType::kEndEvent, instance, time);
  }
  //	getter
  inline const std::unordered_map<SRegst*, float>& regst2ended_at() const {
    return regst2ended_at_;
  }
  inline const NodeMgr<SRegst>& regst_node_mgr() const {
    return regst_node_mgr_;
  }
  inline const ArcMgr<Arc<TaskInstance, SRegst>>& regst_arc_mgr() const {
    return regst_arc_mgr_;
  }
  inline const HasOneArcMgr<Arc<SRegst, SRegstDesc>>& r2rd_arc_mgr() const {
    return r2rd_arc_mgr_;
  }
  inline const std::unordered_map<RegstDescInstance*, SRegst*>&
  regst_desc_instance2regst() const {
    return regst_desc_instance2regst_;
  }
  inline const DeviceInfoProto& device_info_proto() const {
    return device_info_proto_;
  }

  //	setter
  inline ArcMgr<Arc<TaskInstance, SRegst>>& mut_regst_arc_mgr() {
    return regst_arc_mgr_;
  }
  inline HasOneArcMgr<Arc<SRegst, SRegstDesc>>& mut_r2rd_arc_mgr() {
    return r2rd_arc_mgr_;
  }
  inline NodeMgr<SRegst>& mut_regst_node_mgr() { return regst_node_mgr_; }
  inline std::unordered_map<SRegst*, float>& mut_regst2ended_at() {
    return regst2ended_at_;
  }
  inline std::unordered_map<RegstDescInstance*, SRegst*>&
  mut_regst_desc_instance2regst() {
    return regst_desc_instance2regst_;
  }
  inline DeviceInfoProto* mut_device_info_proto() {
    return &device_info_proto_;
  }

  void ForeachNextTaskInstance(TaskInstance* task_instance,
                               const std::function<void(TaskInstance*)>& cb) {
    timenet_arc_mgr().Output(task_instance, cb);
  }

  void ForeachPrevTaskInstance(TaskInstance* task_instance,
                               const std::function<void(TaskInstance*)>& cb) {
    timenet_arc_mgr().Input(task_instance, cb);
  }

 private:
  void EmitEvent(UtilizationEventType event_type, TaskInstance* instance,
                 float time);
  inline const ArcMgr<Arc<TaskInstance>>& timenet_arc_mgr() const {
    return timenet_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskInstance>>& mut_timenet_arc_mgr() {
    return timenet_arc_mgr_;
  }
  void WalkTimeNetReverse(const std::function<void(TaskInstance*)>& cb);
  void WalkFromLossToSink(const std::function<void(TaskInstance*)>& cb);
  void WalkFromLossToSource(const std::function<void(TaskInstance*)>& cb);
  void WalkFromLoss(bool direction,
                    const std::function<void(TaskInstance*)>& cb);
  void LazyRetimingAllNode();
  void EagerRetimingBpNodeWithSplitDeviceHypothesis();
  void LazyRetimingFwNodeWithSplitDeviceHypothesis();

  std::unordered_map<SRegst*, float> regst2ended_at_;
  std::unordered_map<RegstDescInstance*, SRegst*> regst_desc_instance2regst_;
  NodeMgr<SRegst> regst_node_mgr_;
  ArcMgr<Arc<TaskInstance, SRegst>> regst_arc_mgr_;
  HasOneArcMgr<Arc<SRegst, SRegstDesc>> r2rd_arc_mgr_;

  ArcMgr<Arc<TaskInstance>> timenet_arc_mgr_;
  std::unordered_map<SDevice*, TaskInstance*> dev2current_instance_;
  DeviceInfoProto device_info_proto_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_
