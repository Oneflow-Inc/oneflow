#ifndef ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_
#define ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/node.h"

namespace oneflow {
namespace schedule {

typedef Node Regst;

class Batch : public Node {
 public:
  Batch(const std::string name) : Node(name) {}
  Batch() : Node() {}
  virtual ~Batch() {}
};

typedef Arc<Batch, Node> TaskInstance;
typedef Arc<Node> TaskArc;

class Session {
 public:
  Session() = delete;
  virtual ~Session() = default;
  OF_DISALLOW_COPY_AND_MOVE(Session);
  explicit Session(SGraph* graph, uint32_t nr_batch = 2u) : graph_(graph) {
    auto nr_device = graph->DeviceCount();
    auto depth = graph->Depth();
    nr_base_batch_ = std::min(nr_device, depth);
    nr_batch_ = std::max(nr_batch, nr_device * 3);
    NewBatchs();
  }

  void NewBatchs();
  void InitNodeBatchInstance(Node* node);
  std::unique_ptr<std::list<Batch*>> GetBatchNodes();

  inline const SGraph* graph() const { return graph_; }
  inline SGraph* mut_graph() { return graph_; }

  inline const NodeMgr<Batch>& batch_node_mgr() const {
    return batch_node_mgr_;
  }
  inline NodeMgr<Batch>& mut_batch_node_mgr() { return batch_node_mgr_; }

  inline const ArcMgr<TaskInstance>& batch_arc_mgr() const {
    return batch_arc_mgr_;
  }
  inline ArcMgr<TaskInstance>& mut_batch_arc_mgr() { return batch_arc_mgr_; }

  inline uint32_t nr_batch() const { return nr_batch_; }
  inline uint32_t& nr_batch() { return nr_batch_; }
  inline uint32_t nr_base_batch() const { return nr_base_batch_; }
  inline uint32_t& nr_base_batch() { return nr_base_batch_; }

 protected:
  SGraph* graph_;
  uint32_t nr_batch_;
  uint32_t nr_base_batch_;
  NodeMgr<Batch> batch_node_mgr_;
  ArcMgr<TaskInstance> batch_arc_mgr_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_
