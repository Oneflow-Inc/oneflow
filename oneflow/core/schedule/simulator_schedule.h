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
  explicit SimulatorSchedule(const Session& session)
      : Schedule(session),
        event_package_proto_(of_make_unique<UtilizationEventPackageProto>()) {}

  void TimeLinePushBack(const TaskInstance* instance, const SDevice* device);
  void Retiming();
  void InitTimeNet();
  void EmitBeforeRunEvent(const TaskInstance* instance, float time) {
    EmitEvent(UtilizationEventType::kStartEvent, instance, time);
  }
  void EmitAfterRunEvent(const TaskInstance* instance, float time) {
    EmitEvent(UtilizationEventType::kEndEvent, instance, time);
  }
  //	getter
  inline const std::unordered_map<const SRegst*, float>& regst2ended_at()
      const {
    return regst2ended_at_;
  }
  inline const NodeMgr<SRegst>& regst_node_mgr() const {
    return regst_node_mgr_;
  }
  inline const ArcMgr<Arc<TaskInstance, SRegst>>& regst_arc_mgr() const {
    return regst_arc_mgr_;
  }
  inline const ArcMgr<Arc<SRegst, SRegstDesc>>& r2rd_arc_mgr() const {
    return r2rd_arc_mgr_;
  }
  inline const std::unordered_map<const RegstDescInstance*, const SRegst*>&
  regst_desc_instance2regst() const {
    return regst_desc_instance2regst_;
  }
  inline const UtilizationEventPackageProto& event_package_proto() const {
    return *event_package_proto_;
  }
  inline std::unique_ptr<UtilizationEventPackageProto>
  move_event_package_proto() {
    return std::move(event_package_proto_);
  }

  //	setter
  inline ArcMgr<Arc<TaskInstance, SRegst>>* mut_regst_arc_mgr() {
    return &regst_arc_mgr_;
  }
  inline ArcMgr<Arc<SRegst, SRegstDesc>>* mut_r2rd_arc_mgr() {
    return &r2rd_arc_mgr_;
  }
  inline NodeMgr<SRegst>* mut_regst_node_mgr() { return &regst_node_mgr_; }
  inline std::unordered_map<const SRegst*, float>& mut_regst2ended_at() {
    return regst2ended_at_;
  }
  inline std::unordered_map<const RegstDescInstance*, const SRegst*>&
  mut_regst_desc_instance2regst() {
    return regst_desc_instance2regst_;
  }
  inline UtilizationEventPackageProto* mut_device_info_proto() {
    return event_package_proto_.get();
  }

  void ForEachNextTaskInstance(
      const TaskInstance* task_instance,
      const std::function<void(const TaskInstance*)>& cb) {
    timenet_arc_mgr().Output(task_instance, cb);
  }

  void ForEachPrevTaskInstance(
      const TaskInstance* task_instance,
      const std::function<void(const TaskInstance*)>& cb) {
    timenet_arc_mgr().Input(task_instance, cb);
  }

 private:
  void EmitEvent(UtilizationEventType event_type, const TaskInstance* instance,
                 float time);
  inline const ArcMgr<Arc<TaskInstance>>& timenet_arc_mgr() const {
    return timenet_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskInstance>>* mut_timenet_arc_mgr() {
    return &timenet_arc_mgr_;
  }
  void WalkTimeNetReverse(const std::function<void(const TaskInstance*)>& cb);
  void WalkFromLossToSink(const std::function<void(const TaskInstance*)>& cb);
  void WalkFromLossToSource(const std::function<void(const TaskInstance*)>& cb);
  void WalkFromLoss(bool direction,
                    const std::function<void(const TaskInstance*)>& cb);
  void LazyRetimingAllNode();
  void EagerRetimingBpNodeWithSplitDeviceHypothesis();
  void LazyRetimingFwNodeWithSplitDeviceHypothesis();

  std::unordered_map<const SRegst*, float> regst2ended_at_;
  std::unordered_map<const RegstDescInstance*, const SRegst*>
      regst_desc_instance2regst_;
  NodeMgr<SRegst> regst_node_mgr_;
  ArcMgr<Arc<TaskInstance, SRegst>> regst_arc_mgr_;
  ArcMgr<Arc<SRegst, SRegstDesc>> r2rd_arc_mgr_;

  ArcMgr<Arc<TaskInstance>> timenet_arc_mgr_;
  std::unordered_map<const SDevice*, const TaskInstance*> dev2current_instance_;
  std::unique_ptr<UtilizationEventPackageProto> event_package_proto_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_
