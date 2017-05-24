#ifndef ONEFLOW_GRAPH_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_COMP_TASK_NODE_H_

#include <algorithm>
#include "graph/task_node.h"

namespace oneflow {

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() {}
  virtual ~CompTaskNode() = default;

  // Getters and Setters
  uint64_t parallel_id() const { return parallel_id_; }
  void set_parallel_id(uint64_t parallel_id) { parallel_id_ = parallel_id; }
  bool IsLossNode() const { return chain_node()->IsLossNode(); }
  std::string VisualStr() const override;
  virtual void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
    ret->set_parallel_id(parallel_id_);
  }
  std::string device_name() const;

  // Build Exec and Set Produced Regsts
  void DataFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void DataFwInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  void MdUpdtFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void MdUpdtFwInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  void MdLoadFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void MdLoadFwInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  void MdSaveFwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void MdSaveFwInferShapeOfBlobsInProducedRegsts(TaskGraph*);

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    auto fw_comp_code = of_dynamic_cast<CompTaskNode*> (fw_node);
    parallel_id_ = fw_comp_code->parallel_id_;
  }

 private:
  using Lbn2NodeBnMap =
      HashMap<std::string, std::pair<ExecNode*, std::string>>;
  
  OVERRIDE_IF_FW_BP_FOR_FUNC(BuildExecAndEnrollLbn2Regsts);
  OVERRIDE_IF_FW_BP_FOR_FUNC(InferShapeOfBlobsInProducedRegsts);

  void FwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph);
  void FwInferShapeOfBlobsInProducedRegsts(TaskGraph* gph);
  void FwBuildFromUserOps(
      Lbn2NodeBnMap* lbn2producer,
      Lbn2NodeBnMap* extern_in_lbn2consumer);
  void FwSetExecNodeFromInRegst(
      const Lbn2NodeBnMap& extern_in_lbn2consumer);
  void FwEnrollLbn2OutRegst(const Lbn2NodeBnMap& lbn2producer);
  void FwEnrollLbn2OutRegstWhenLoss();
  void FwEnrollLbn2OutRegstWhenNotLoss(const Lbn2NodeBnMap& lbn2producer);
  void FwEnrollLbn2ActivationRegst();
  void FwEnrollLbn2ModelAndTmpRegsts();

  void BpBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void BpInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  void BpBuildExecGraph();
  void BpEnrollLbn2ProducedRegst();
  void BpEnrollLbn2ActivationDiffRegst();
  void BpSetExecNodeFromOutDiffRegst();
  void BpEnrollLbn2InDiffRegst();
  void BpEnrollLbn2ModelDiffRegst();

  uint64_t parallel_id_;

};

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec);

class HostCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostCompTaskNode);
  HostCompTaskNode() = default;
  ~HostCompTaskNode() = default;

  void ToProto(TaskProto* ret) const override {
    CompTaskNode::ToProto(ret);
  }

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<HostCompTaskNode> ();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }
  TaskType task_type() const override { return HostCompTask; }

};

class DeviceCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTaskNode);
  DeviceCompTaskNode() = default;
  ~DeviceCompTaskNode() = default;
  
  void ToProto(TaskProto* ret) const override {
    CompTaskNode::ToProto(ret);
  };

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<DeviceCompTaskNode> ();
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    CompTaskNode::InitWithFwNode(fw_node);
  }
  TaskType task_type() const override { return DeviceCompTask; }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMP_TASK_NODE_H_
