#ifndef ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_
#define ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

typedef SNode SRegst;

class Batch : public SNode {
 public:
  Batch(const std::string name) : SNode(name) {}
  Batch() : SNode() {}
  virtual ~Batch() {}
};

typedef Arc<Batch, STask> TaskInstance;
typedef Arc<Batch, SRegstDesc> RegstDescInstance;
typedef Arc<Batch, TaskArc> TaskArcInstance;

class Session {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Session);
  Session() = delete;
  virtual ~Session() = default;
  explicit Session(const SGraph& graph, const UtilizationGraph& ugraph,
                   uint32_t nr_base_batch)
      : sgraph_(&graph),
        ugraph_(&ugraph),
        nr_base_batch_(nr_base_batch),
        nr_batch_(nr_base_batch * 3) {
    NewBatchs();
  }
  explicit Session(const SGraph& graph, const UtilizationGraph& ugraph)
      : sgraph_(&graph), ugraph_(&ugraph) {
    uint32_t nr_device = graph.DeviceCount();
    uint32_t depth = graph.Depth();
    nr_base_batch_ = std::min(nr_device + 3u, depth);
    nr_base_batch_ = std::max(nr_base_batch_, 12u);
    nr_batch_ = nr_base_batch_ * 3;
    NewBatchs();
  }

  void NewBatchs();
  const Batch* EndBatch() const {
    return batch_node_mgr().Find(nr_batch() - 1);
  }
  TaskInstance* GetNextBatchInstance(TaskInstance* instance) const {
    return GetNextBatchInstance(instance, 1);
  }
  TaskInstance* GetNextBatchInstance(TaskInstance* instance,
                                     int32_t step) const;
  TaskInstance* GetPrevBatchInstance(TaskInstance* instance) const;
  std::unique_ptr<std::list<Batch*>> GetBatchNodes() const;

  inline const SGraph* sgraph() const { return sgraph_; }

  inline const NodeMgr<Batch>& batch_node_mgr() const {
    return batch_node_mgr_;
  }
  inline NodeMgr<Batch>& mut_batch_node_mgr() { return batch_node_mgr_; }

  inline const ArcMgr<TaskInstance>& task_instance_mgr() const {
    return task_instance_mgr_;
  }
  inline ArcMgr<TaskInstance>& mut_task_instance_mgr() {
    return task_instance_mgr_;
  }

  inline const ArcMgr<RegstDescInstance>& regst_desc_instance_mgr() const {
    return regst_desc_instance_mgr_;
  }
  inline ArcMgr<RegstDescInstance>& mut_regst_desc_instance_mgr() {
    return regst_desc_instance_mgr_;
  }

  inline const ArcMgr<TaskArcInstance>& task_arc_instance_mgr() const {
    return task_arc_instance_mgr_;
  }
  inline ArcMgr<TaskArcInstance>& mut_task_arc_instance_mgr() {
    return task_arc_instance_mgr_;
  }

  inline uint32_t nr_batch() const { return nr_batch_; }
  inline uint32_t nr_base_batch() const { return nr_base_batch_; }
  inline uint32_t nr_stable_batch() const { return nr_base_batch_; }
  inline uint32_t nr_unstable_batch() const { return nr_base_batch_; }

  inline uint32_t& mut_nr_batch() { return nr_batch_; }
  inline uint32_t& mut_nr_base_batch() { return nr_base_batch_; }

 protected:
  const SGraph* sgraph_;
  const UtilizationGraph* ugraph_;
  uint32_t nr_base_batch_;
  uint32_t nr_batch_;
  NodeMgr<Batch> batch_node_mgr_;
  ArcMgr<TaskInstance> task_instance_mgr_;
  ArcMgr<RegstDescInstance> regst_desc_instance_mgr_;
  ArcMgr<TaskArcInstance> task_arc_instance_mgr_;
};

template<uint32_t tpl_nr_base_batch>
class FixedBatchSession final : public Session {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FixedBatchSession);
  FixedBatchSession() = delete;
  virtual ~FixedBatchSession() = default;
  explicit FixedBatchSession(const SGraph& graph,
                             const UtilizationGraph& ugraph)
      : Session(graph, ugraph, std::max(tpl_nr_base_batch, 2u)) {}
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_
