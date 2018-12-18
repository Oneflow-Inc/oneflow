#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class OpEdge;

class OpNode final : public Node<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpNode);
  explicit OpNode(const ParallelDesc& parallel_desc, const OperatorConf& op_conf)
      : parallel_desc_(parallel_desc),
        op_(ConstructOp(op_conf, parallel_desc.device_type())),
        ibns_(op_->input_bns().begin(), op_->input_bns().end()),
        has_in_diff_(false) {}
  ~OpNode() = default;

  const Operator& op() const { return *op_; }
  bool HasBackward() const { return has_in_diff() || has_model_diff(); }
  bool has_in_diff() const { return has_in_diff_; }
  bool has_model_diff() const { return op().model_diff_bns().size() > 0; }
  void set_has_in_diff(bool has_in_diff) { has_in_diff_ = has_in_diff; }
  const ParallelDesc& parallel_desc() const { return parallel_desc_; }

  BlobDesc* BlobDesc4BnInOp(const std::string& bn_in_op);

 private:
  BlobDesc* MutBlobDesc(const LogicalBlobId& lbi);

  ParallelDesc parallel_desc_;
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  bool has_in_diff_;
  HashMap<LogicalBlobId, std::shared_ptr<BlobDesc>> lbi2blob_desc_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(const std::vector<LogicalBlobId>& lbis,
                  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns)
      : lbis_(lbis), lbi2ibns_(lbi2ibns) {}
  ~OpEdge() = default;

  const std::vector<LogicalBlobId>& lbis() const { return lbis_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return lbi2ibns_; }

 private:
  std::vector<LogicalBlobId> lbis_;
  HashMap<LogicalBlobId, std::vector<std::string>> lbi2ibns_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  explicit OpGraph(const JobDesc* job_desc) : job_desc_(job_desc) { Init(); }
  ~OpGraph() = default;

  void InferOpModelSize(HashMap<std::string, size_t>* op_name2model_size);

 private:
  void Init();
  void InitNodes();
  void InitEdges();
  void UpdateOpNodeHasInDiff();
  void InferNodeBlobDesc() const;

  const JobDesc* job_desc_;
  HashMap<std::string, OpNode*> op_name2op_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_H_
