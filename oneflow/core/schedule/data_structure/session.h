#ifndef ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_
#define ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/data_structure/node.h"

namespace oneflow {
namespace schedule {

class Session {
 public:
  Session() = delete;
  virtual ~Session() = default;
  OF_DISALLOW_COPY_AND_MOVE(Session);
  explicit Session(GraphNode* graph, uint32_t nr_batch = 2u) : graph_(graph) {
    auto nr_device = graph->DeviceCount();
    auto depth = graph->Depth();
    nr_base_batch_ = std::min(nr_device, depth);
    nr_batch_ = std::max(nr_batch, nr_device * 3);
    NewBatchs();
  }

  void NewBatchs();
  void InitNodeBatchInstance(Node* node);
  std::unique_ptr<std::list<Node*>> GetBatchNodes();

  inline const GraphNode* graph() const { return graph_; }
  inline GraphNode* mut_graph() { return graph_; }

  inline const NodeMgr<Node>& batch_node_mgr() const { return batch_node_mgr_; }
  inline NodeMgr<Node>& mut_batch_node_mgr() { return batch_node_mgr_; }

  inline const ArcMgr& batch_arc_mgr() const { return batch_arc_mgr_; }
  inline ArcMgr& mut_batch_arc_mgr() { return batch_arc_mgr_; }

  inline const NodeMgr<Node>& batch_instance_node_mgr() const {
    return batch_instance_node_mgr_;
  }
  inline NodeMgr<Node>& mut_batch_instance_node_mgr() {
    return batch_instance_node_mgr_;
  }
  inline uint32_t nr_batch() const { return nr_batch_; }
  inline uint32_t& nr_batch() { return nr_batch_; }
  inline uint32_t nr_base_batch() const { return nr_base_batch_; }
  inline uint32_t& nr_base_batch() { return nr_base_batch_; }

 protected:
  GraphNode* graph_;
  uint32_t nr_batch_;
  uint32_t nr_base_batch_;
  NodeMgr<Node> batch_instance_node_mgr_;
  NodeMgr<Node> batch_node_mgr_;
  ArcMgr batch_arc_mgr_;
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_DATA_STRUCTURE_SESSION_H_
