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
        op_(ConstructOp(op_conf, parallel_desc.device_type(), &GlobalJobDesc())),
        ibns_(op_->input_bns().begin(), op_->input_bns().end()) {}
  ~OpNode() = default;

  // Getters
  const Shape* GetInputBlobFastestTimeShape() const;
  const Shape* GetInputOutputFastestTimeShape() const;
  const Shape* out_blob_time_shape() const;
  bool IsTimeShapeIdentity() const;
  const Operator& op() const { return *op_; }
  const ParallelDesc& parallel_desc() const { return parallel_desc_; }
  const SbpSignature& sbp_signature() const { return sbp_signature_; }
  const SbpParallel& SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  const SbpParallel& SbpParallel4BnInOp(const std::string& bn_in_op) const;
  const BlobDesc& LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const;
  const OptInt64& BatchAxis4Lbi(const LogicalBlobId& lbi) const;
  const OpNode& ProducerOpNode4Lbi(const LogicalBlobId& lbi) const;
  const OpNode& SrcNode4InputBnInOp(const std::string& bn_in_op) const;
  const OpNode& ProducerOpNode4BnInOp(const std::string& bn_in_op) const;
  const ParallelDesc& BlobParallelDesc4Obn(const std::string& obn) const;

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
  HashMap<std::string, std::vector<std::shared_ptr<BlobDesc>>>* mut_bn2parallel_id2blob_desc() {
    return &bn2parallel_id2blob_desc_;
  }
  OptInt64* MutBatchAxis4Lbi(const LogicalBlobId& lbi);
  BlobDesc* MutLogicalBlobDesc4Lbi(const LogicalBlobId& lbi);
  OpNode* MutSrcNode4InputBnInOp(const std::string& bn_in_op) const;
  OpNode* MutProducerOpNode4BnInOp(const std::string& bn_in_op);
  OpNode* MutSrcNode4InputLbi(const LogicalBlobId& lbi) const;
  OpNode* MutProducerOpNode4Lbi(const LogicalBlobId& lbi);
  void ForEachSplitOrBroadcastBlobDesc(const BlobDesc& blob_desc, const SbpParallel& sbp_parallel,
                                       const std::function<void(const BlobDesc&)>& Handler) const;

  int64_t GetAxisParallelNum(
      const std::function<void(bool*, int32_t*, int64_t*)>& GetAxisParallelInfo) const;
  void ConcatBlobDesc(const ParallelDesc& blob_parallel_desc,
                      const std::vector<std::shared_ptr<BlobDesc>>& blob_descs,
                      const SbpParallel& sbp_parallel, BlobDesc* concatenated_blob_desc) const;
  void SplitLogicalInputBlobDesc();
  void ConcatLogicalOutputBlobDesc();
  void CheckBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;
  void InferBlobParallelDesc();

  ParallelDesc parallel_desc_;
  HashMap<std::string, ParallelDesc> obn2blob_parallel_desc_;
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  std::unique_ptr<Shape> out_blob_time_shape_;
  SbpSignature sbp_signature_;
  HashMap<LogicalBlobId, OptInt64> lbi2batch_axis_;
  HashMap<std::string, std::vector<std::shared_ptr<BlobDesc>>> bn2parallel_id2blob_desc_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(const std::vector<LogicalBlobId>& lbis,
                  const HashMap<LogicalBlobId, std::string>& lbi2obn,
                  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns)
      : lbis_(lbis), lbi2obn_(lbi2obn), lbi2ibns_(lbi2ibns) {}
  ~OpEdge() = default;

  void InitDistributeHierarchyInfo();

  // Getters
  const std::vector<LogicalBlobId>& lbis() const { return lbis_; }
  const HashMap<LogicalBlobId, std::string>& lbi2obn() const { return lbi2obn_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return lbi2ibns_; }
  std::string VisualStr() const override;
  bool is_strict_121() const { return is_strict_121_; }

 private:
  void InitIsStrict121();
  bool CalcIsStrict121Connected() const;

  std::vector<LogicalBlobId> lbis_;
  HashMap<LogicalBlobId, std::string> lbi2obn_;
  HashMap<LogicalBlobId, std::vector<std::string>> lbi2ibns_;

  bool is_strict_121_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  explicit OpGraph(const Job& job) { Init(job); }
  ~OpGraph() = default;

  const OpNode* OpNode4OpName(const std::string& name) const;

  std::function<const BlobDesc&(const LogicalBlobId&)> MakeGetterBlobDesc4ModelLbi() const;

  int32_t GetModelSplitAxis(const std::string& op_name, const LogicalBlobId& lbi) const;
  BalancedSplitter GetBalancedSplitter(const std::string& op_name, const LogicalBlobId& lbi) const;
  int64_t GetParallelNum(const std::string& op_name) const;
  int64_t GetSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const;
  const SbpParallel& GetSbpParallel(const std::string& op_name, const LogicalBlobId& lbi) const;
  DataType GetBlobDataType(const LogicalBlobId& lbi) const;
  const BlobDesc& GetLogicalBlobDesc(const LogicalBlobId& lbi) const;
  void CheckBlobDescs(const std::string& op_name,
                      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;

  // a set of nodes is called a chain family if they can divided into several connected chains
  void ForEachChainFamily(const std::function<void(const HashSet<OpNode*>&)>& Handler) const;

  std::function<bool(const std::string&, const std::string&)>
  MakePredicatorIsOpNameDataOrCtrlReachable() const;

  void ForEachDataAndCtrlInNode(OpNode* node, const std::function<void(OpNode*)>& Handler) const;
  void ForEachDataAndCtrlOutNode(OpNode* node, const std::function<void(OpNode*)>& Handler) const;

  void DumpLogicalBlobDesc(JobBuilder* job_builder) const;
  void DumpSbpSignature(JobBuilder* job_builder) const;
  void DumpOpTimeShape(JobBuilder* job_builder) const;
  void DumpBatchAxisLbi(JobBuilder* job_builder) const;

 private:
  void Init(const Job& job);
  void InitNodes(const Job& job);
  void InitEdges();
  void InitProducerOpName2CtrlConsumerOpNames(const Job& job);
  void CheckIsDAG() const;
  void InferTimeShape() const;
  void InferOpNodeSbpSignature(OpNode* op_node, const SbpSignature& sbp_sig_conf) const;
  void InferOpNodeLogicalBlobDesc(OpNode* op_node) const;
  void InferLogicalBlobDesc(const Job& job) const;
  bool IsBatchAxisBlob(const std::string& op_name, const LogicalBlobId& lbi) const;
  std::string GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const;
  LogicalBlobId GetLogicalBlobIdKey(const std::string& op_name, const LogicalBlobId& lbi) const;

  std::function<bool(const OpNode*, const OpNode*)> MakePredicatorIsDataOrCtrlReachable() const;
  std::list<OpNode*> DataOrCtrlSourceNodes() const;

  HashMap<std::string, OpNode*> op_name2op_node_;
  HashMap<std::string, HashSet<std::string>> producer_op_name2ctrl_consumer_op_names_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
