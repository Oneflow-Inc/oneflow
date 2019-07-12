#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
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
  const Shape* GetInputBlobFastestTimeShape() const;
  const Shape* out_blob_time_shape() const;
  const Operator& op() const { return *op_; }
  bool HasBackward() const { return has_in_diff() || has_model_diff(); }
  bool has_in_diff() const { return has_in_diff_; }
  bool has_model_diff() const { return op().model_diff_bns().size() > 0; }
  void set_has_in_diff(bool has_in_diff) { has_in_diff_ = has_in_diff; }
  const ParallelDesc& parallel_desc() const { return parallel_desc_; }
  const SbpSignature& sbp_signature() const { return sbp_signature_; }
  const SbpParallel& SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  const SbpParallel& SbpParallel4BnInOp(const std::string& bn_in_op) const;
  const BlobDesc& LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const;
  bool HasBatchDim4Lbi(const LogicalBlobId& lbi) const;
  const OpNode* ProducerOpNode4Lbi(const LogicalBlobId& lbi) const;

  std::string VisualStr() const override;

 private:
  friend class OpGraph;
  friend class OpEdge;
  // Getters
  const Shape* GetInputBlobTimeShape(const std::string& bn_in_op) const;

  // Setters
  ParallelDesc* mut_parallel_desc() { return &parallel_desc_; }
  SbpSignature* mut_sbp_signature() { return &sbp_signature_; }
  Shape* mut_out_blob_time_shape();
  HashMap<std::string, std::vector<BlobDesc>>* mut_bn2parallel_id2blob_desc() {
    return &bn2parallel_id2blob_desc_;
  }
  bool* MutHasBatchDim4Lbi(const LogicalBlobId& lbi);
  BlobDesc* MutLogicalBlobDesc4Lbi(const LogicalBlobId& lbi);
  OpNode* MutSrcNode4InputBnInOp(const std::string& bn_in_op) const;
  OpNode* MutProducerOpNode4BnInOp(const std::string& bn_in_op);
  OpNode* MutSrcNode4InputLbi(const LogicalBlobId& lbi) const;
  OpNode* MutProducerOpNode4Lbi(const LogicalBlobId& lbi);

  int64_t GetAxisParallelNum(
      const std::function<void(bool*, int32_t*, int64_t*)>& GetAxisParallelInfo) const;
  void ConcatBlobDesc(const std::vector<BlobDesc>& blob_descs, const SbpParallel& sbp_parallel,
                      BlobDesc* concatenated_blob_desc) const;
  void SplitLogicalInputBlobDesc();
  void ConcatLogicalOutputBlobDesc();
  void CheckBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;

  ParallelDesc parallel_desc_;
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  bool has_in_diff_;
  std::unique_ptr<Shape> out_blob_time_shape_;
  SbpSignature sbp_signature_;
  HashMap<LogicalBlobId, bool> lbi2has_batch_dim_;
  HashMap<std::string, std::vector<BlobDesc>> bn2parallel_id2blob_desc_;
  HashMap<LogicalBlobId, BlobDesc> lbi2logical_blob_desc_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(const std::vector<LogicalBlobId>& lbis,
                  const HashMap<LogicalBlobId, std::string>& lbi2obn,
                  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns)
      : lbis_(lbis), lbi2obn_(lbi2obn), lbi2ibns_(lbi2ibns), has_diff_(false) {}
  ~OpEdge() = default;

  const std::vector<LogicalBlobId>& lbis() const { return lbis_; }
  const HashMap<LogicalBlobId, std::string>& lbi2obn() const { return lbi2obn_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return lbi2ibns_; }
  bool has_diff() const { return has_diff_; }
  std::string VisualStr() const override;

  void set_has_diff(bool val) { has_diff_ = val; }

 private:
  std::vector<LogicalBlobId> lbis_;
  HashMap<LogicalBlobId, std::string> lbi2obn_;
  HashMap<LogicalBlobId, std::vector<std::string>> lbi2ibns_;
  bool has_diff_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  explicit OpGraph(const Job& job) { Init(job); }
  ~OpGraph() = default;

  void InferOpModelSize(HashMap<std::string, size_t>* op_name2model_size);
  std::function<const BlobDesc&(const LogicalBlobId&)> MakeGetterBlobDesc4ModelLbi() const;

  int32_t GetModelSplitAxis(const std::string& op_name, const LogicalBlobId& lbi) const;
  BalancedSplitter GetBalancedSplitter(const std::string& op_name, const LogicalBlobId& lbi) const;
  const SbpParallel& GetSbpParallel(const std::string& op_name, const LogicalBlobId& lbi) const;
  DataType GetBlobDataType(const LogicalBlobId& lbi) const;
  const BlobDesc& GetLogicalBlobDesc(const LogicalBlobId& lbi) const;
  void CheckBlobDescs(const std::string& op_name,
                      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;

  // a set of nodes is called a pseudo chain if they can merge into a chain regardless of the
  // connections before their source nodes
  void ForEachPseudoChain(const std::function<void(const HashSet<OpNode*>&)>& Handler) const;
  // a set of nodes is called a chain family if they can divided into several connected chains
  void ForEachChainFamily(const std::function<void(const HashSet<OpNode*>&)>& Handler) const;

  std::function<bool(const LogicalBlobId&, const std::string&)>
  MakePredicatorIsLbiAllConsumersReachableToOpName() const;

 private:
  void Init(const Job& job);
  void InitNodes(const Job& job);
  void InitEdges();
  void InitProducerOpName2CtrlConsumerOpNames(const Job& job);
  void CheckIsDAG() const;
  void FixOpParallelDesc() const;
  void UpdateOpNodeHasInDiff() const;
  void InferTimeShape() const;
  void InferOpNodeSbpSignature(OpNode* op_node, const SbpSignature& sbp_sig_conf) const;
  void InferOpNodeLogicalBlobDesc(OpNode* op_node) const;
  void InferLogicalBlobDesc(const Job& job) const;
  bool IsBatchDimBlob(const std::string& op_name, const LogicalBlobId& lbi) const;
  std::string GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const;
  LogicalBlobId GetLogicalBlobIdKey(const std::string& op_name, const LogicalBlobId& lbi) const;
  void ForEachPseudoChain(const HashSet<OpNode*>& nodes,
                          const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable,
                          const std::function<void(const HashSet<OpNode*>&)>& Handler) const;
  void ReverseTopoGetPseudoChain(
      const HashSet<OpNode*>& op_nodes, HashSet<OpNode*>* chain,
      const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable) const;

  std::function<bool(const OpNode*, const OpNode*)> MakePredicatorIsDataOrCtrlReachable() const;
  void ForEachDataAndCtrlInNode(OpNode* node, const std::function<void(OpNode*)>& Handler) const;
  void ForEachDataAndCtrlOutNode(OpNode* node, const std::function<void(OpNode*)>& Handler) const;

  int64_t GetSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const;
  HashMap<std::string, OpNode*> op_name2op_node_;
  HashMap<std::string, HashSet<std::string>> producer_op_name2ctrl_consumer_op_names_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
