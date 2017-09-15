#ifndef ONEFLOW_CORE_SCHEDULE_SGRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_SGRAPH_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/schedule/snode.h"

namespace oneflow {
namespace schedule {

//	static schedule device
class SDevice : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SDevice);
  SDevice() = default;
  ~SDevice() = default;
  SDevice(std::string name, float bandwidth = 1.0)
      : SNode(name), bandwidth_(bandwidth) {}

  float time() const { return 1 / bandwidth_; }
  float bandwith() const { return bandwidth_; }
  float delay() const { return delay_; }
  uint64_t memory_limit() const { return memory_limit_; }

  float& mut_bandwidth() { return bandwidth_; }
  float& mut_delay() { return delay_; }
  uint64_t& mut_memory_limit() { return memory_limit_; }

  void set_time(float t) { bandwidth_ = 1 / t; }

 private:
  float bandwidth_ = 1;
  float delay_ = 0;
  uint64_t memory_limit_ = ULLONG_MAX;
};

//	static schedule task

class STask : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(STask);
  explicit STask(const std::string name) : SNode(name) {}
  STask(const std::string name, float wl) : SNode(name), workload_(wl) {}

  STask() {}
  virtual ~STask() {}

  inline float workload() const { return workload_; }
  inline uint32_t depth() const { return depth_; }
  inline const SDevice& device() const { return *device_; }
  inline bool has_device() const { return device_ != nullptr; }

  inline float& mut_workload() { return workload_; }
  inline uint32_t& mut_depth() { return depth_; }
  inline SDevice*& mut_device() { return device_; }

 protected:
  uint32_t depth_;
  float workload_ = 1.0;
  SDevice* device_;
};

typedef Arc<STask> TaskArc;

class SRegstDesc : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SRegstDesc);
  SRegstDesc(const std::string name) : SNode(name) {}
  SRegstDesc() : SNode() {}
  virtual ~SRegstDesc() {}

  inline uint64_t regst_memory_size() const { return regst_memory_size_; }
  inline const STask& owner_task() const { return *owner_task_; }
  inline uint32_t min_regst_count() const { return min_regst_count_; }
  inline uint32_t origin_regst_count() const { return origin_regst_count_; }

  inline uint64_t& mut_regst_memory_size() { return regst_memory_size_; }
  inline STask*& mut_owner_task() { return owner_task_; }
  inline uint32_t& mut_min_regst_count() { return min_regst_count_; }
  inline uint32_t& mut_origin_regst_count() { return origin_regst_count_; }

 private:
  uint64_t regst_memory_size_ = 1u;
  uint32_t min_regst_count_ = 0u;
  uint32_t origin_regst_count_ = 2u;
  STask* owner_task_;
};

//	static schedule graph

class SGraph : public SNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraph);
  SGraph() = default;
  virtual ~SGraph() = default;

  explicit SGraph(const Plan& plan) : SNode("plan-graph"), plan_(&plan) {
    InitSourceAndSink();
  }

  std::string ToDotString();

  static bool DescNodeOrder(STask* a, STask* b) {
    return a->depth() < b->depth();
  }

  static bool AscNodeOrder(STask* a, STask* b) {
    return a->depth() > b->depth();
  }

  void ForEachNext(STask* node, const std::function<void(STask*)>& cb) const {
    arc_mgr().Output(node, cb);
  }

  void ForEachPrev(STask* node, const std::function<void(STask*)>& cb) const {
    arc_mgr().Input(node, cb);
  }

  void ForEachNode(const std::function<void(STask*)>& cb) const;
  void ForEachNode(const std::function<void(const STask&)>& cb) const;
  void MutForEachChild(const std::function<void(STask*)>& cb) const;
  void ForEachChild(const std::function<void(const STask&)>& cb) const;
  void ForEachAscendant(STask*, const std::function<void(STask*)>& cb) const;
  void ForEachDescendant(STask*, const std::function<void(STask*)>& cb) const;
  void ForEachRegstDesc(const std::function<void(SRegstDesc*)>& cb) const;

  void Walk(const std::function<void(STask*)>& cb) const;
  void WalkArc(const std::function<void(Arc<STask>*)>& cb) const;
  uint32_t DeviceCount() const;
  uint32_t Depth() const;
  void WalkReverse(const std::function<void(STask*)>& cb) const;
  void WalkArcReverse(const std::function<void(Arc<STask>*)>& cb) const;
  void ForEachArc(const std::function<void(Arc<STask>*)>& cb) const;
  uint32_t LossNodes(std::list<STask*>* l) const;

  //	getter
  inline const Plan& plan() const { return *plan_; }
  STask* source() const { return source_; }
  STask* sink() const { return sink_; }
  inline const NodeMgr<STask>& node_mgr() const { return node_mgr_; }
  inline const ArcMgr<Arc<STask>>& arc_mgr() const { return arc_mgr_; }
  inline const HasOneArcMgr<Arc<STask, SDevice>>& device_arc_mgr() const {
    return device_arc_mgr_;
  }
  inline const ArcMgr<Arc<SGraph, STask>>& loss_arc_mgr() const {
    return loss_arc_mgr_;
  }
  inline const ArcMgr<Arc<STask>>& ascendant_arc_mgr() const {
    return ascendant_arc_mgr_;
  }
  inline const ArcMgr<Arc<SGraph, STask>>& children_arc_mgr() const {
    return children_arc_mgr_;
  }
  inline const NodeMgr<SRegstDesc>& regst_desc_mgr() const {
    return regst_desc_mgr_;
  }
  inline const ArcMgr<Arc<STask, SRegstDesc>>& produced_regst_desc_mgr() const {
    return produced_regst_desc_mgr_;
  }
  inline const ArcMgr<Arc<STask, SRegstDesc>>& subscribed_regst_desc_mgr()
      const {
    return subscribed_regst_desc_mgr_;
  }

  //	setter
  STask*& mut_source() { return source_; }
  STask*& mut_sink() { return sink_; }
  inline NodeMgr<STask>* mut_node_mgr() { return &node_mgr_; }
  inline NodeMgr<STask>* mut_fake_node_mgr() { return &fake_node_mgr_; }
  inline ArcMgr<Arc<STask>>* mut_arc_mgr() { return &arc_mgr_; }
  inline HasOneArcMgr<Arc<STask, SDevice>>* mut_device_arc_mgr() {
    return &device_arc_mgr_;
  }
  inline NodeMgr<SDevice>* mut_device_mgr() { return &device_mgr_; }
  inline ArcMgr<Arc<SGraph, STask>>* mut_loss_arc_mgr() {
    return &loss_arc_mgr_;
  }
  inline ArcMgr<Arc<STask>>* mut_ascendant_arc_mgr() {
    return &ascendant_arc_mgr_;
  }
  inline ArcMgr<Arc<SGraph, STask>>* mut_children_arc_mgr() {
    return &children_arc_mgr_;
  }
  inline NodeMgr<SRegstDesc>* mut_regst_desc_mgr() { return &regst_desc_mgr_; }
  inline ArcMgr<Arc<STask, SRegstDesc>>* mut_produced_regst_desc_mgr() {
    return &produced_regst_desc_mgr_;
  }
  inline ArcMgr<Arc<STask, SRegstDesc>>* mut_subscribed_regst_desc_mgr() {
    return &subscribed_regst_desc_mgr_;
  }

 protected:
  void Update() {
    UpdateSourceAndSink();
    InitDepth();
    InitAscendantArc();
    RemoveUselessArc();
    UpdateTask();
    UpdateRegstDesc();
  }
  void InitAscendantArc();
  void UpdateSourceAndSink();
  void InitSourceAndSink();
  void InitDepth();
  void UpdateTask();
  void UpdateRegstDesc();
  void RemoveUselessArc();
  bool ReachableWithoutArc(const TaskArc* arc) const;

 private:
  const Plan* plan_;
  STask* source_;
  STask* sink_;
  NodeMgr<STask> node_mgr_;
  NodeMgr<STask> fake_node_mgr_;
  NodeMgr<SRegstDesc> regst_desc_mgr_;
  NodeMgr<SDevice> device_mgr_;
  ArcMgr<TaskArc> arc_mgr_;
  ArcMgr<Arc<SGraph, STask>> loss_arc_mgr_;
  ArcMgr<Arc<SGraph, STask>> children_arc_mgr_;
  ArcMgr<Arc<STask>> ascendant_arc_mgr_;
  ArcMgr<Arc<STask, SRegstDesc>> produced_regst_desc_mgr_;
  ArcMgr<Arc<STask, SRegstDesc>> subscribed_regst_desc_mgr_;
  HasOneArcMgr<Arc<STask, SDevice>> device_arc_mgr_;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SGRAPH_H_
