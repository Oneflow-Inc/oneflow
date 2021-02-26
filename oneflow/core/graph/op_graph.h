/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/mirrored_parallel.pb.h"
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
  std::shared_ptr<const Operator> shared_op() const { return op_; }
  const ParallelDesc& parallel_desc() const { return parallel_desc_; }
  const SbpSignature& sbp_signature() const { return *CHECK_JUST(op().sbp_signature()); }
  const SbpParallel& SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  const SbpParallel& SbpParallel4BnInOp(const std::string& bn_in_op) const;
  const BlobDesc& LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const;
  const OpNode& ProducerOpNode4Lbi(const LogicalBlobId& lbi) const;
  const OpNode& SrcNode4Ibn(const std::string& bn_in_op) const;
  const ParallelDesc& BlobParallelDesc4Obn(const std::string& obn) const;

  std::string VisualStr() const override;

 private:
  friend class OpGraph;
  friend class OpEdge;
  // Getters
  const Shape* GetInputBlobTimeShape(const std::string& bn_in_op) const;

  // Setters
  Operator* mut_op() { return op_.get(); }
  ParallelDesc* mut_parallel_desc() { return &parallel_desc_; }
  Shape* mut_out_blob_time_shape();
  HashMap<std::string, std::vector<std::shared_ptr<BlobDesc>>>* mut_bn2parallel_id2blob_desc() {
    return &bn2parallel_id2blob_desc_;
  }
  BlobDesc* MutLogicalBlobDesc4Lbi(const LogicalBlobId& lbi);
  OpNode* MutSrcNode4Ibn(const std::string& bn_in_op) const;
  OpNode* MutSrcNode4InputLbi(const LogicalBlobId& lbi) const;
  void ForEachSplitOrBroadcastBlobDesc(const BlobDesc& blob_desc, const SbpParallel& sbp_parallel,
                                       const std::function<void(const BlobDesc&)>& Handler) const;

  void ConcatBlobDesc(const ParallelDesc& blob_parallel_desc,
                      const std::vector<std::shared_ptr<BlobDesc>>& blob_descs,
                      const SbpParallel& sbp_parallel, BlobDesc* concatenated_blob_desc) const;
  void SplitLogicalInputBlobDesc();
  void ConcatLogicalOutputBlobDesc();
  void CheckBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const;
  void InferBlobParallelDesc();
  void InitLbi2SourceNode();
  void InitInputBlobFastestTimeShape();
  void InitLbi2SbpParallel();

  ParallelDesc parallel_desc_;
  HashMap<std::string, ParallelDesc> obn2blob_parallel_desc_;
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  std::unique_ptr<Shape> out_blob_time_shape_;
  HashMap<std::string, std::vector<std::shared_ptr<BlobDesc>>> bn2parallel_id2blob_desc_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  HashMap<LogicalBlobId, OpNode*> lbi2source_node_;
  std::unique_ptr<Shape> input_blob_fastest_time_shape_;
  HashMap<LogicalBlobId, SbpParallel> lbi2sbp_parallel_;
  std::unique_ptr<Shape> parallel_hierarchy_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(std::shared_ptr<std::vector<LogicalBlobId>> lbis,
                  std::shared_ptr<HashMap<LogicalBlobId, std::string>> lbi2obn,
                  std::shared_ptr<HashMap<LogicalBlobId, std::vector<std::string>>> lbi2ibns)
      : lbis_(std::move(lbis)), lbi2obn_(std::move(lbi2obn)), lbi2ibns_(std::move(lbi2ibns)) {}
  ~OpEdge() override = default;

  void InitDistributeHierarchyInfo();

  // Getters
  const std::vector<LogicalBlobId>& lbis() const { return *lbis_; }
  const HashMap<LogicalBlobId, std::string>& lbi2obn() const { return *lbi2obn_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return *lbi2ibns_; }
  std::string VisualStr() const override;
  bool is_strict_121() const { return is_strict_121_; }

 private:
  void InitIsStrict121();
  bool CalcIsStrict121Connected() const;

  std::shared_ptr<std::vector<LogicalBlobId>> lbis_;
  std::shared_ptr<HashMap<LogicalBlobId, std::string>> lbi2obn_;
  std::shared_ptr<HashMap<LogicalBlobId, std::vector<std::string>>> lbi2ibns_;

  bool is_strict_121_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  explicit OpGraph(const Job& job) { CHECK_JUST(Init(job)); }
  explicit OpGraph() = default;
  ~OpGraph() override = default;

  static Maybe<OpGraph> New(const Job& job);

  Maybe<void> ForEachOpNode(const std::function<Maybe<void>(const OpNode&)>& DoEach) const;

  const OpNode* OpNode4OpName(const std::string& name) const;

  int64_t GetParallelNum(const std::string& op_name) const;
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
  // NOTE(chengcheng): For topo for each with ctrl edges. OpEdge is ONLY data edge.
  std::list<OpNode*> DataOrCtrlSourceNodes() const;

  void DumpLogicalBlobDesc(Job* job) const;
  void DumpSbpSignature(Job* job) const;
  void DumpOpTimeShape(Job* job) const;

  Maybe<void> Init(const Job& job);

 private:
  void InitNodes(const Job& job);
  void InitEdges();
  void InitProducerOpName2CtrlConsumerOpNames(const Job& job);
  void CheckIsDAG() const;
  void InferBlobLastUsed() const;
  void InferTimeShape() const;
  void InferOpNodeSbpSignature(OpNode* op_node, const SbpSignature& sbp_sig_conf) const;
  Maybe<void> InferOpNodeMirroredSignature(OpNode* op_node, bool is_mirrored_conf) const;
  Maybe<void> InferOpNodeLogicalBlobDesc(OpNode* op_node) const;
  Maybe<void> InferLogicalBlobDesc(const Job& job) const;
  std::string GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const;
  LogicalBlobId GetLogicalBlobIdKey(const std::string& op_name, const LogicalBlobId& lbi) const;

  std::function<bool(const OpNode*, const OpNode*)> MakePredicatorIsDataOrCtrlReachable() const;

  HashMap<std::string, OpNode*> op_name2op_node_;
  std::list<std::string> op_names_;
  HashMap<std::string, HashSet<std::string>> producer_op_name2ctrl_consumer_op_names_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
