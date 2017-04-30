#ifndef ONEFLOW_GRAPH_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_COMP_TASK_NODE_H_

#include <algorithm>
#include "graph/task_node.h"

namespace oneflow {

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  // Getters and Setters
  uint64_t parallel_id() const { return parallel_id_; }
  void set_parallel_id(uint64_t parallel_id) { parallel_id_ = parallel_id; }
  bool IsLossNode() const { return chain_node()->IsLossNode(); }
  bool IsFaker() const { return chain_node()->IsFaker(); }
  std::string VisualStr() const override;

  // Build Exec and Set Produced Regsts
  void DataFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void DataFwInferShape4LbnInProducedRegsts(TaskGraph*);
  void MdUpdtFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void MdUpdtFwInferShape4LbnInProducedRegsts(TaskGraph*);
  void MdLoadFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void MdLoadFwInferShape4LbnInProducedRegsts(TaskGraph*);
  void MdSaveFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void MdSaveFwInferShape4LbnInProducedRegsts(TaskGraph*);

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    parallel_id_ = of_dynamic_cast<CompTaskNode*> (fw_node)->parallel_id_;
  }

 private:
  using Lbn2NodeBnMap =
      HashMap<std::string, std::pair<ExecNode*, std::string>>;
  
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override {
    if (IsFwNode()) {
      return FwBuildExecAndEnrollLbn2Regsts(gph);
    } else {
      return BpBuildExecAndEnrollLbn2Regsts(gph);
    }
  }
  void InferShape4LbnInProducedRegsts(TaskGraph* gph) override {
    if (IsFwNode()) {
      return FwInferShape4LbnInProducedRegsts(gph);
    } else {
      return BpInferShape4LbnInProducedRegsts(gph);
    }
  }

  void FwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override {
    (this->*(gph->Func4FwBuildExecAndEnrollLbn2Regsts()))(gph);
  }
  void FwInferShape4LbnInProducedRegsts(TaskGraph* gph) override {
    (this->*(gph->Func4FwInferShape4LbnInProducedRegsts()))(gph);
  }
  void FwBuildFromUserOps(
      Lbn2NodeBnMap* lbn2producer,
      Lbn2NodeBnMap* extern_in_lbn2consumer);
  void FwSetExecNodeFromInRegst(
      const Lbn2NodeBnMap& extern_in_lbn2consumer);
  void FwEnrollLbn2OutRegst(const Lbn2NodeBnMap& lbn2producer);
  void FwEnrollLbn2ActivationRegst();
  void FwEnrollLbn2ModelAndTmpRegsts();

  void BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) override;
  void BpInferShape4LbnInProducedRegsts(TaskGraph*) override;
  void BpBuildExecGraph(
      const ExecGraph& fw_gph,
      HashMap<const ExecNode*, ExecNode*>* fw_node2bp_node,
      HashMap<ExecEdge*, const ExecEdge*>* bp_edge2fw_edge);
  void BpEnrollLbn2ProducedRegst(
      const HashMap<const ExecNode*, ExecNode*>& fw_node2bp_node,
      const HashMap<ExecEdge*, const ExecEdge*>& bp_edge2fw_edge);

  uint64_t parallel_id_;

};

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec);

class HostCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompTaskNode);
  HostCompTaskNode() = default;
  ~HostCompTaskNode() = default;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<HostCompTaskNode> ();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }

};

class DeviceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTaskNode);
  DeviceCompTaskNode() = default;
  ~DeviceCompTaskNode() = default;
  
 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<DeviceCompTaskNode> ();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMP_TASK_NODE_H_
