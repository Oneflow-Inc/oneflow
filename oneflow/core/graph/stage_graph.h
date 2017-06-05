#ifndef ONEFLOW_CORE_GRAPH_STAGE_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_STAGE_GRAPH_H_

#include "oneflow/core/common/range.h"
#include "oneflow/core/graph/chain_graph.h"

namespace oneflow {

class StageEdge;

class StageNode final : public Node<StageNode, StageEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StageNode);
  StageNode() = default;
  ~StageNode() = default;

  std::string machine_id_str() const {
    return std::to_string(machine_id_);
  }
  const uint64_t& machine_id() const {
    return machine_id_;
  }
  uint64_t& mut_machine_id() {
    return machine_id_;
  }

  const ChainNode* chain_node() const {
    return chain_node_;
  }
  void set_chain_node(const ChainNode* new_chain_node) {
    chain_node_ = new_chain_node;
  }

  const Range& parallel_range() const {
    return parallel_range_;
  }
  Range& mut_parallel_range() {
    return parallel_range_;
  }

  const std::vector<uint64_t>& SortedDevicePhyIds() const {
    return chain_node_->parallel_desc()->sorted_device_phy_ids(machine_id_);
  }

  std::string VisualStr() const override {
    return machine_id_str() + "\\n" + chain_node_->VisualStr();
  }

 private:
  const ChainNode* chain_node_;
  uint64_t machine_id_;
  Range parallel_range_;

};

class StageEdge final : public Edge<StageNode, StageEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StageEdge);
  StageEdge() = default;
  ~StageEdge() = default;
    
 private:
};

class StageGraph final : public Graph<StageNode, StageEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StageGraph);
  StageGraph() = delete;
  ~StageGraph() = default;

  StageGraph(std::unique_ptr<const ChainGraph>&& chain_gph,
             const std::string& dot_filepath);

  const ChainGraph* chain_gph() const { return chain_gph_.get(); }

 private:
  std::unique_ptr<const ChainGraph> chain_gph_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_STAGE_GRAPH_H_
