#ifndef ONEFLOW_CORE_SCHEDULE_SGRAPH_H_
#define ONEFLOW_CORE_SCHEDULE_SGRAPH_H_

#include "oneflow/core/common/preprocessor.h"
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

  STask() = default;
  virtual ~STask() = default;

  inline float workload() const { return workload_; }
  inline uint32_t depth() const { return depth_; }
  inline const SDevice& device() const { return *device_; }
  inline bool has_device() const { return device_ != nullptr; }

  inline float& mut_workload() { return workload_; }
  inline uint32_t& mut_depth() { return depth_; }
  inline const SDevice*& mut_device() { return device_; }

 protected:
  uint32_t depth_;
  float workload_ = 1.0;
  const SDevice* device_;
};

class EmptyTask : public STask {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmptyTask);
  explicit EmptyTask(const std::string name) : STask(name) {}
  ~EmptyTask() = default;
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
  inline const STask*& mut_owner_task() { return owner_task_; }
  inline uint32_t& mut_min_regst_count() { return min_regst_count_; }
  inline uint32_t& mut_origin_regst_count() { return origin_regst_count_; }

 private:
  uint64_t regst_memory_size_ = 1u;
  uint32_t min_regst_count_ = 0u;
  uint32_t origin_regst_count_ = 2u;
  const STask* owner_task_;
};

//	static schedule graph

class SGraph : public SNode {
#define SGRAPH_NODE_SEQ           \
  OF_PP_MAKE_TUPLE_SEQ(EmptyTask) \
  OF_PP_MAKE_TUPLE_SEQ(STask)     \
  OF_PP_MAKE_TUPLE_SEQ(SDevice)   \
  OF_PP_MAKE_TUPLE_SEQ(SRegstDesc)

#define SGRAPH_ARC_SEQ                                               \
  OF_PP_MAKE_TUPLE_SEQ(arc_mgr, STask, STask)                        \
  OF_PP_MAKE_TUPLE_SEQ(ascendant_arc_mgr, STask, STask)              \
  OF_PP_MAKE_TUPLE_SEQ(loss_arc_mgr, SGraph, STask)                  \
  OF_PP_MAKE_TUPLE_SEQ(children_arc_mgr, SGraph, STask)              \
  OF_PP_MAKE_TUPLE_SEQ(produced_regst_desc_mgr, STask, SRegstDesc)   \
  OF_PP_MAKE_TUPLE_SEQ(subscribed_regst_desc_mgr, STask, SRegstDesc) \
  OF_PP_MAKE_TUPLE_SEQ(device_arc_mgr, STask, SDevice)

 public:
  OF_DISALLOW_COPY_AND_MOVE(SGraph);
  SGraph() = default;
  virtual ~SGraph() = default;

  explicit SGraph(const Plan& plan) : SNode("sgraph"), plan_(&plan) {
    InitSourceAndSink();
  }

  std::string ToDotString();

  static bool DescNodeOrder(const STask* a, const STask* b) {
    return a->depth() < b->depth();
  }

  static bool AscNodeOrder(const STask* a, const STask* b) {
    return a->depth() > b->depth();
  }

  void ForEachNext(const STask* node,
                   const std::function<void(const STask*)>& cb) const {
    arc_mgr().Output(node, cb);
  }

  void ForEachPrev(const STask* node,
                   const std::function<void(const STask*)>& cb) const {
    arc_mgr().Input(node, cb);
  }

  void ForEachNode(const std::function<void(const STask*)>& cb) const;
  void ForEachNode(const std::function<void(const STask&)>& cb) const;
  void MutForEachChild(const std::function<void(const STask*)>& cb) const;
  void ForEachChild(const std::function<void(const STask&)>& cb) const;
  void ForEachAscendant(STask*,
                        const std::function<void(const STask*)>& cb) const;
  void ForEachDescendant(STask*,
                         const std::function<void(const STask*)>& cb) const;
  void ForEachRegstDesc(const std::function<void(const SRegstDesc*)>& cb) const;

  void Walk(const std::function<void(const STask*)>& cb) const;
  void WalkArc(const std::function<void(const Arc<STask>*)>& cb) const;
  uint32_t DeviceCount() const;
  uint32_t Depth() const;
  void WalkReverse(const std::function<void(const STask*)>& cb) const;
  void WalkArcReverse(const std::function<void(const Arc<STask>*)>& cb) const;
  void ForEachArc(const std::function<void(const Arc<STask>*)>& cb) const;
  uint32_t LossNodes(std::list<const STask*>* l) const;

  //	getter
  inline const Plan& plan() const { return *plan_; }
  const STask* source() const { return source_; }
  const STask* sink() const { return sink_; }

  template<typename node_type>
  inline const NodeMgr<node_type>& node_mgr() const;

#define SGRAPH_ARC_MGR_GETTER(arc_mgr_field, src_node_type, dst_node_type) \
  inline const ArcMgr<Arc<src_node_type, dst_node_type>>& arc_mgr_field()  \
      const {                                                              \
    return OF_PP_CAT(arc_mgr_field, _);                                    \
  }
  OF_PP_FOR_EACH_TUPLE(SGRAPH_ARC_MGR_GETTER, SGRAPH_ARC_SEQ);

  //	setter
  const STask*& mut_source() { return source_; }
  const STask*& mut_sink() { return sink_; }

  template<typename node_type>
  inline NodeMgr<node_type>* mut_node_mgr();

#define SGRAPH_ARC_MGR_SETTER(arc_mgr_field, src_node_type, dst_node_type) \
  inline ArcMgr<Arc<src_node_type, dst_node_type>>* OF_PP_CAT(             \
      mut_, arc_mgr_field)() {                                             \
    return &OF_PP_CAT(arc_mgr_field, _);                                   \
  }
  OF_PP_FOR_EACH_TUPLE(SGRAPH_ARC_MGR_SETTER, SGRAPH_ARC_SEQ);

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
  const STask* source_;
  const STask* sink_;

#define SGRAPH_NODE_MGR_MEMBER(node_type) \
  NodeMgr<node_type> OF_PP_CAT(node_type, _node_mgr_);
  OF_PP_FOR_EACH_TUPLE(SGRAPH_NODE_MGR_MEMBER, SGRAPH_NODE_SEQ);

#define SGRAPH_ARC_MGR_MEMBER(arc_mgr_field, src_node_type, dst_node_type) \
  ArcMgr<Arc<src_node_type, dst_node_type>> OF_PP_CAT(arc_mgr_field, _);
  OF_PP_FOR_EACH_TUPLE(SGRAPH_ARC_MGR_MEMBER, SGRAPH_ARC_SEQ);
};

#define SGRAPH_NODE_MGR_GETTER(node_type)                                \
  template<>                                                             \
  inline const NodeMgr<node_type>& SGraph::node_mgr<node_type>() const { \
    return OF_PP_CAT(node_type, _node_mgr_);                             \
  }
OF_PP_FOR_EACH_TUPLE(SGRAPH_NODE_MGR_GETTER, SGRAPH_NODE_SEQ);

#define SGRAPH_NODE_MGR_SETTER(node_type)                        \
  template<>                                                     \
  inline NodeMgr<node_type>* SGraph::mut_node_mgr<node_type>() { \
    return &OF_PP_CAT(node_type, _node_mgr_);                    \
  }
OF_PP_FOR_EACH_TUPLE(SGRAPH_NODE_MGR_SETTER, SGRAPH_NODE_SEQ);

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_SGRAPH_H_
