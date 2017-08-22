/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_

#include <limits.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/schedule/schedule_engine.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"

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

  void ForeachNextTaskInstance(TaskInstance* task_instance,
                               const std::function<void(TaskInstance*)>& cb) {
    timenet_arc_mgr().Output(task_instance, cb);
  }

  void ForeachPrevTaskInstance(TaskInstance* task_instance,
                               const std::function<void(TaskInstance*)>& cb) {
    timenet_arc_mgr().Input(task_instance, cb);
  }

 protected:
  inline const ArcMgr<Arc<TaskInstance>>& timenet_arc_mgr() const {
    return timenet_arc_mgr_;
  }
  inline ArcMgr<Arc<TaskInstance>>& mut_timenet_arc_mgr() {
    return timenet_arc_mgr_;
  }
  void WalkTimeNetReverse(const std::function<void(TaskInstance*)>& cb);
  void WalkBpTimeNet(const std::function<void(TaskInstance*)>& cb);

  std::unordered_map<SRegst*, float> regst2ended_at_;
  std::unordered_map<RegstDescInstance*, SRegst*> regst_desc_instance2regst_;
  NodeMgr<SRegst> regst_node_mgr_;
  ArcMgr<Arc<TaskInstance, SRegst>> regst_arc_mgr_;
  HasOneArcMgr<Arc<SRegst, SRegstDesc>> r2rd_arc_mgr_;

  ArcMgr<Arc<TaskInstance>> timenet_arc_mgr_;
  std::unordered_map<SDevice*, TaskInstance*> dev2current_instance_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_SCHEDULE_H_
