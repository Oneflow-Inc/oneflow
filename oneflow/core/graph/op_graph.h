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
  explicit OpNode(const std::shared_ptr<const ParallelDesc>& parallel_desc,
                  const OperatorConf& op_conf);
  ~OpNode() = default;

  // Getters
  bool IsTimeShapeIdentity() const;
  const Operator& op() const { return *op_; }
  std::shared_ptr<const Operator> shared_op() const { return op_; }
  const ParallelDesc& parallel_desc() const { return *parallel_desc_; }
  const SbpSignature& sbp_signature() const { return *CHECK_JUST(op().sbp_signature()); }
  const ParallelDistributionSignature& parallel_distribution_signature() const {
    return *CHECK_JUST(op().parallel_distribution_signature());
  }
  const SbpParallel& SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  const SbpParallel& SbpParallel4BnInOp(const std::string& bn_in_op) const;
  const ParallelDistribution& ParallelDistribution4Lbi(const LogicalBlobId& lbi) const;
  const ParallelDistribution& ParallelDistribution4BnInOp(const std::string& bn_in_op) const;
  const BlobDesc& LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const;
  const OpNode& ProducerOpNode4Lbi(const LogicalBlobId& lbi) const;
  const OpNode& SrcNode4Ibn(const std::string& bn_in_op) const;

  std::string VisualStr() const override;

 private:
  friend class OpGraph;
  friend class OpEdge;

  // Setters
  Operator* mut_op() { return op_.get(); }
  OpNode* MutSrcNode4Ibn(const std::string& bn_in_op) const;
  OpNode* MutSrcNode4InputLbi(const LogicalBlobId& lbi) const;
  void InitLbi2SourceNode();
  void InitLbi2ParallelDistribution();

  std::shared_ptr<const ParallelDesc> parallel_desc_;
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  HashMap<LogicalBlobId, OpNode*> lbi2source_node_;
  HashMap<LogicalBlobId, ParallelDistribution> lbi2parallel_distribution_;
  std::vector<std::pair<const OpNode*, int32_t>> input_index2producer_and_output_index_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(std::shared_ptr<std::vector<LogicalBlobId>> lbis,
                  std::shared_ptr<HashMap<LogicalBlobId, std::string>> lbi2obn,
                  std::shared_ptr<HashMap<LogicalBlobId, std::vector<std::string>>> lbi2ibns)
      : lbis_(std::move(lbis)), lbi2obn_(std::move(lbi2obn)), lbi2ibns_(std::move(lbi2ibns)) {}
  ~OpEdge() override = default;

  // Getters
  const std::vector<LogicalBlobId>& lbis() const { return *lbis_; }
  const HashMap<LogicalBlobId, std::string>& lbi2obn() const { return *lbi2obn_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return *lbi2ibns_; }
  std::string VisualStr() const override;

 private:
  std::shared_ptr<std::vector<LogicalBlobId>> lbis_;
  std::shared_ptr<HashMap<LogicalBlobId, std::string>> lbi2obn_;
  std::shared_ptr<HashMap<LogicalBlobId, std::vector<std::string>>> lbi2ibns_;
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
  const ParallelDistribution& GetParallelDistribution(const std::string& op_name,
                                                      const LogicalBlobId& lbi) const;
  DataType GetBlobDataType(const LogicalBlobId& lbi) const;
  const BlobDesc& GetLogicalBlobDesc(const LogicalBlobId& lbi) const;

  std::function<bool(const std::string&, const std::string&)>
  MakePredicatorIsOpNameDataOrCtrlReachable() const;

  void ForEachDataAndCtrlInNode(OpNode* node, const std::function<void(OpNode*)>& Handler) const;
  void ForEachDataAndCtrlOutNode(OpNode* node, const std::function<void(OpNode*)>& Handler) const;
  // NOTE(chengcheng): For topo for each with ctrl edges. OpEdge is ONLY data edge.
  std::list<OpNode*> DataOrCtrlSourceNodes() const;

  void DumpLogicalBlobDesc(Job* job) const;
  void DumpArgSignature(Job* job) const;
  void DumpParallelDistributionSignature(Job* job) const;

  Maybe<void> Init(const Job& job);

 private:
  void InitNodes(const Job& job);
  void InitEdges();
  void InitProducerOpName2CtrlConsumerOpNames(const Job& job);
  void CheckIsDAG() const;
  void InferBlobLastUsed() const;
  void InferTimeShape() const;
  void InferOpNodeParallelDistributionSignature(
      OpNode* op_node, const ParallelDistributionSignature& parallel_distribution_sig_conf) const;
  Maybe<void> InferOpNodeMirroredSignature(OpNode* op_node, bool is_mirrored_conf) const;
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
