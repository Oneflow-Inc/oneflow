#ifndef ONEFLOW_CORE_GRAPH_EXEC_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_EXEC_GRAPH_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/exec_sequence.pb.h"
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class ExecNode;

class ExecEdge final : public Edge<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecEdge);
  ExecEdge() = default;
  ~ExecEdge() = default;

  // Getters
  const LogicalBlobId& lbi() const { return lbi_; }
  const std::string& src_bn() const { return src_bn_; }
  const std::string& dst_bn() const { return dst_bn_; }

  // Setters
  void set_lbi(const LogicalBlobId& lbi) { lbi_ = lbi; }
  std::string& mut_src_bn() { return src_bn_; }
  std::string& mut_dst_bn() { return dst_bn_; }

 private:
  // various names for one blob
  LogicalBlobId lbi_;
  std::string src_bn_;
  std::string dst_bn_;
};

class ExecNode final : public Node<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecNode);
  ExecNode() : fw_node_(nullptr) {}
  ~ExecNode() = default;

  std::shared_ptr<const Operator> op() const { return op_; }
  std::shared_ptr<const Operator>& mut_op() { return op_; }
  RegstDesc* RegstDesc4BnInOp(const std::string& bn) const { return bn_in_op2regst_.at(bn).get(); }

  void BindBnWithRegst(const std::string& bn, std::shared_ptr<RegstDesc>);
  void BindBnsWithRegst(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                        std::shared_ptr<RegstDesc>);
  void AddBnToRegstAndBindIt(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                             std::shared_ptr<RegstDesc>);
  void BindBnWithOneOfTheRegsts(const std::string&, const std::list<std::shared_ptr<RegstDesc>>&);
  void UnbindBnWithEmptyRegst();

  void set_fw_node(ExecNode* val) { fw_node_ = val; }
  ExecNode* fw_node() { return fw_node_; }

  std::string VisualStr() const override { return op_->op_name(); }
  void ToProto(bool is_forward, const ParallelContext*, ExecNodeProto*) const;

  void InferBlobDescs(const ParallelContext* parallel_ctx);

 private:
  const OpContext* op_context() const { return fw_node_ ? fw_node_->op_ctx_.get() : op_ctx_.get(); }
  std::function<const BlobDesc&(const std::string&)> GetLogicalBlobDesc4BnInOpFunc() const;
  std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOpFunc() const;

  std::shared_ptr<const Operator> op_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> bn_in_op2regst_;
  ExecNode* fw_node_;

  std::unique_ptr<OpContext> op_ctx_;
};

class ExecGraph final : public Graph<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecGraph);
  ExecGraph() = default;
  ~ExecGraph() = default;

  void ToExecSequence(bool is_forward, const ParallelContext*, ExecSequence*) const;
  const char* TypeName() const override { return "ExecGraph"; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_EXEC_GRAPH_H_
