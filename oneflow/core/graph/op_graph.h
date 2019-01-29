#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/blob_parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class OpEdge;
class OpGraph;

class OpNode final : public Node<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpNode);
  explicit OpNode(const ParallelDesc& parallel_desc, const OperatorConf& op_conf)
      : parallel_desc_(parallel_desc),
        op_(ConstructOp(op_conf, parallel_desc.device_type())),
        ibns_(op_->input_bns().begin(), op_->input_bns().end()),
        has_in_diff_(false) {}
  ~OpNode() = default;

  // Getters
  const Shape& out_blob_time_shape() const { return out_blob_time_shape_; }
  const Operator& op() const { return *op_; }
  bool HasBackward() const { return has_in_diff() || has_model_diff(); }
  bool has_in_diff() const { return has_in_diff_; }
  bool has_model_diff() const { return op().model_diff_bns().size() > 0; }
  void set_has_in_diff(bool has_in_diff) { has_in_diff_ = has_in_diff; }
  const ParallelDesc& parallel_desc() const { return parallel_desc_; }
  const BlobDesc& NoParallelBlobDesc4Lbi(const LogicalBlobId& lbi) const;
  const BlobDesc& LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const;
  const BlobParallelDesc& BlobParallelDesc4BnInOp(const std::string& bn) const;
  const BlobParallelDesc& BlobParallelDesc4Lbi(const LogicalBlobId& lbi) const;
  const LogicalBlobParallelDesc& Lbpd4Lbi(const LogicalBlobId& lbi) const;
  const Shape* GetInputBlobTimeShape(const std::string& bn_in_op) const;
  const Shape* GetInputBlobTimeShape() const;

  std::string VisualStr() const override;

 private:
  friend class OpGraph;
  // Setters
  ParallelDesc* mut_parallel_desc() { return &parallel_desc_; }
  Shape* mut_out_blob_time_shape() { return &out_blob_time_shape_; }
  HashMap<std::string, std::vector<BlobDesc>>* mut_bn2parallel_id2blob_desc() {
    return &bn2parallel_id2blob_desc_;
  }
  BlobDesc* NoParallelBlobDesc4BnInOp(const std::string& bn_in_op);
  BlobDesc* MutNoParallelBlobDesc(const LogicalBlobId& lbi);
  BlobDesc* LogicalBlobDesc4BnInOp(const std::string& bn_in_op);
  BlobDesc* MutLogicalBlobDesc(const LogicalBlobId& lbi);
  BlobParallelDesc* MutBlobParallelDesc4BnInOp(const std::string& bn_in_op,
                                               int32_t model_split_axis);
  LogicalBlobParallelDesc* MutLbpd4Lbi(const LogicalBlobId& lbi);
  OpNode* SrcNode4InputBnInOp(const std::string& bn_in_op) const;
  OpNode* ProducerOpNode4BnInOp(const std::string& bn_in_op);
  OpNode* SrcNode4InputLbi(const LogicalBlobId& lbi) const;
  OpNode* ProducerOpNode4Lbi(const LogicalBlobId& lbi);
  void ForEachParallelBlobDesc(
      const BlobDesc& blob_desc,
      const std::function<void(bool*, int32_t*, int64_t*)>& GetAxisParallelInfo,
      const std::function<void(const BlobDesc&)>& Handler) const;
  int64_t GetAxisParallelNum(
      const std::function<void(bool*, int32_t*, int64_t*)>& GetAxisParallelInfo) const;
  void ConcatBlobDesc(const std::vector<BlobDesc>& blob_descs,
                      const std::function<void(bool*, int32_t*, int64_t*)>& GetAxisParallelInfo,
                      BlobDesc* concatenated_blob_desc) const;
  void SplitLogicalInputBlobDesc();
  void ConcatLogicalOutputBlobDesc();
  void CheckBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;

  ParallelDesc parallel_desc_;
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  bool has_in_diff_;
  Shape out_blob_time_shape_;
  HashMap<LogicalBlobId, BlobDesc> lbi2no_parallel_blob_desc_;
  HashMap<LogicalBlobId, BlobDesc> lbi2logical_blob_desc_;
  HashMap<std::string, std::vector<BlobDesc>> bn2parallel_id2blob_desc_;
  HashMap<LogicalBlobId, BlobParallelDesc> lbi2blob_parallel_desc_;
  HashMap<LogicalBlobId, LogicalBlobParallelDesc> lbi2lbpd_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(const std::vector<LogicalBlobId>& lbis,
                  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns)
      : lbis_(lbis), lbi2ibns_(lbi2ibns), has_diff_(false) {}
  ~OpEdge() = default;

  const std::vector<LogicalBlobId>& lbis() const { return lbis_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return lbi2ibns_; }
  bool has_diff() const { return has_diff_; }
  std::string VisualStr() const override;

  void set_has_diff(bool val) { has_diff_ = val; }

 private:
  std::vector<LogicalBlobId> lbis_;
  HashMap<LogicalBlobId, std::vector<std::string>> lbi2ibns_;
  bool has_diff_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  explicit OpGraph(const JobDesc* job_desc) : job_desc_(job_desc) { Init(); }
  ~OpGraph() = default;

  void InferOpModelSize(HashMap<std::string, size_t>* op_name2model_size);

  int32_t GetModelSplitAxis(const std::string& op_name, const LogicalBlobId& lbi) const;
  BalancedSplitter GetDataBalancedSplitter(const std::string& op_name, const LogicalBlobId& lbi,
                                           const ParallelDesc& parallel_desc) const;
  BalancedSplitter GetModelBalancedSplitter(const std::string& op_name, const LogicalBlobId& lbi,
                                            const ParallelDesc& parallel_desc) const;
  const BlobParallelDesc& GetBlobParallelDesc(const std::string& op_name,
                                              const LogicalBlobId& lbi) const;
  void CheckBlobDescs(const std::string& op_name,
                      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;

  // a set of nodes is called a pseudo chain if they can merge into a chain regardless of the
  // connections before their source nodes
  void ForEachPseudoChain(const std::function<void(const HashSet<OpNode*>&)>& Handler) const;

 private:
  void Init();
  void InitNodes();
  void InitEdges();
  void FixOpParallelDesc() const;
  void UpdateOpNodeHasInDiff() const;
  void InferTimeShape() const;
  void InferNoParallelBlobDesc() const;
  void InferModelSplitAxis(
      HashMap<OpNode*, HashMap<LogicalBlobId, int32_t>>* op_node2lbi2model_split_axis) const;
  void InferLogicalBlobParallelDesc(
      const HashMap<OpNode*, HashMap<LogicalBlobId, int32_t>>& op_node2lbi2model_split_axis) const;
  void InferLogicalBlobDesc() const;
  std::string GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const;
  LogicalBlobId GetLogicalBlobIdKey(const std::string& op_name, const LogicalBlobId& lbi) const;
  void ForEachPseudoChain(const std::vector<OpNode*>& nodes,
                          const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable,
                          const std::function<void(const HashSet<OpNode*>&)>& Handler) const;
  void ReverseTopoGetPseudoChain(
      const HashSet<OpNode*>& op_nodes, HashSet<OpNode*>* chain,
      const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable) const;
  std::function<bool(OpNode* src, OpNode* dst)> MakePredicatorIsReachable() const;
  void ForEachComponentWithSameDataParallelDescAndTimeShape(
      const std::function<void(const std::vector<OpNode*>&)>& Handler) const;

  int64_t GetModelSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const;
  int64_t GetDataSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const;
  int64_t GetParallelNum(const std::string& op_name, const LogicalBlobId& lbi) const;
  const JobDesc* job_desc_;
  HashMap<std::string, OpNode*> op_name2op_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
