/**
 * Copyright 2017 Xinqi Li
 */
#ifndef ONEFLOW_CORE_SCHEDULE_SGRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_SGRAPH_H_

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
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/snode.h"
#include "oneflow/core/schedule/util.h"

namespace oneflow {
namespace schedule {

//	static schedule device
class SDevice : public SNode {
 public:
  SDevice(std::string name, unsigned int time) : SNode(name), time_(time) {}
  unsigned int time() const { return time_; }
  unsigned int& mut_time() { return time_; }

  uint64_t memory_limit() const { return memory_limit_; }
  uint64_t& mut_memory_limit() { return memory_limit_; }

  DEFINE_METHOD_TYPE();

 private:
  unsigned int time_;
  uint64_t memory_limit_ = ULLONG_MAX;
};

//	static schedule task

class STask : public SNode {
 public:
  explicit STask(const std::string name) : SNode(name) {}
  STask() {}
  virtual ~STask() {}
  DEFINE_METHOD_TYPE();
  inline int depth() const { return depth_; }
  inline const SDevice* device() const { return device_; }

  inline int& mut_depth() { return depth_; }
  inline SDevice*& mut_device() { return device_; }

 protected:
  int depth_;
  SDevice* device_;
};

class SRegstDesc : public SNode {
 public:
  SRegstDesc(const std::string name) : SNode(name) {}
  SRegstDesc() : SNode() {}
  virtual ~SRegstDesc() {}

  inline uint64_t regst_memory_size() const { return regst_memory_size_; }
  inline const STask* owner_task() const { return owner_task_; }

  inline uint64_t& mut_regst_memory_size() { return regst_memory_size_; }
  inline STask*& mut_owner_task() { return owner_task_; }

 private:
  uint64_t regst_memory_size_ = 1u;
  STask* owner_task_;
};

//	static schedule graph

class SGraph : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraph);
  SGraph() = default;
  ~SGraph() = default;
  DEFINE_METHOD_TYPE();

  explicit SGraph(std::string name) : SNode(name) { InitSourceAndSink(); }
  explicit SGraph(const Plan& plan) : SNode("plan-graph") {
    InitSourceAndSink();
  }

  static bool DescNodeOrder(STask* a, STask* b) {
    return a->depth() < b->depth();
  }

  static bool AscNodeOrder(STask* a, STask* b) {
    return a->depth() > b->depth();
  }

  void InitAscendentArc();

  void ForeachNode(const std::function<void(STask*)>& cb) const;
  void ForeachAscendent(STask*, const std::function<void(STask*)>& cb) const;
  void ForeachDescendent(STask*, const std::function<void(STask*)>& cb) const;
  void ForeachNodeWithSourceAndSink(
      const std::function<void(STask*)>& cb) const;
  void ForeachRegstDesc(const std::function<void(SRegstDesc*)>& cb) const;

  void Walk(const std::function<void(STask*)>& cb);
  void WalkArc(const std::function<void(Arc<STask>*)>& cb);
  uint32_t DeviceCount() const;
  uint32_t Depth() const;
  void WalkReverse(const std::function<void(STask*)>& cb);
  void WalkArcReverse(const std::function<void(Arc<STask>*)>& cb);
  void ForeachArc(const std::function<void(Arc<STask>*)>& cb) const;
  int LossNodes(std::list<STask*>* l) const;
  STask* source() const { return source_; }
  STask*& mut_source() { return source_; }

  STask* sink() const { return sink_; }
  STask*& mut_sink() { return sink_; }

  inline const NodeMgr<STask>& node_mgr() const { return node_mgr_; }
  inline NodeMgr<STask>& mut_node_mgr() { return node_mgr_; }

  inline NodeMgr<STask>& mut_fake_node_mgr() { return fake_node_mgr_; }

  inline const ArcMgr<Arc<STask>>& arc_mgr() const { return arc_mgr_; }
  inline ArcMgr<Arc<STask>>& mut_arc_mgr() { return arc_mgr_; }

  inline const HasOneArcMgr<Arc<STask, SDevice>>& device_arc_mgr() const {
    return device_arc_mgr_;
  }
  inline HasOneArcMgr<Arc<STask, SDevice>>& mut_device_arc_mgr() {
    return device_arc_mgr_;
  }

  inline NodeMgr<SDevice>& mut_device_mgr() { return device_mgr_; }

  inline const ArcMgr<Arc<SGraph, STask>>& loss_arc_mgr() const {
    return loss_arc_mgr_;
  }
  inline ArcMgr<Arc<SGraph, STask>>& mut_loss_arc_mgr() {
    return loss_arc_mgr_;
  }

  inline const ArcMgr<Arc<STask>>& ascendent_arc_mgr() const {
    return ascendent_arc_mgr_;
  }
  inline ArcMgr<Arc<STask>>& mut_ascendent_arc_mgr() {
    return ascendent_arc_mgr_;
  }

  inline const ArcMgr<Arc<SGraph, STask>>& children_arc_mgr() const {
    return children_arc_mgr_;
  }
  inline ArcMgr<Arc<SGraph, STask>>& mut_children_arc_mgr() {
    return children_arc_mgr_;
  }

  inline NodeMgr<SRegstDesc>& mut_regst_desc_mgr() { return regst_desc_mgr_; }

  inline const ArcMgr<Arc<STask, SRegstDesc>>& produced_regst_desc_mgr() const {
    return produced_regst_desc_mgr_;
  }
  inline ArcMgr<Arc<STask, SRegstDesc>>& mut_produced_regst_desc_mgr() {
    return produced_regst_desc_mgr_;
  }

  inline const ArcMgr<Arc<STask, SRegstDesc>>& subscribed_regst_desc_mgr()
      const {
    return subscribed_regst_desc_mgr_;
  }
  inline ArcMgr<Arc<STask, SRegstDesc>>& mut_subscribed_regst_desc_mgr() {
    return subscribed_regst_desc_mgr_;
  }

 protected:
  void Update() {
    UpdateSourceAndSink();
    InitDepth();
    InitAscendentArc();
    UpdateTask();
    UpdateRegstDesc();
  }
  void UpdateSourceAndSink();
  void InitSourceAndSink();
  void InitDepth();
  void UpdateTask();
  void UpdateRegstDesc();

 private:
  STask* source_;
  STask* sink_;
  NodeMgr<STask> node_mgr_;
  NodeMgr<STask> fake_node_mgr_;
  NodeMgr<SRegstDesc> regst_desc_mgr_;
  NodeMgr<SDevice> device_mgr_;
  ArcMgr<Arc<STask>> arc_mgr_;
  ArcMgr<Arc<SGraph, STask>> loss_arc_mgr_;
  ArcMgr<Arc<SGraph, STask>> children_arc_mgr_;
  ArcMgr<Arc<STask>> ascendent_arc_mgr_;
  ArcMgr<Arc<STask, SRegstDesc>> produced_regst_desc_mgr_;
  ArcMgr<Arc<STask, SRegstDesc>> subscribed_regst_desc_mgr_;
  HasOneArcMgr<Arc<STask, SDevice>> device_arc_mgr_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SGRAPH_H_
