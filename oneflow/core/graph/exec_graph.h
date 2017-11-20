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
  const std::string& lbn() const { return lbn_; }
  const std::string& src_bn() const { return src_bn_; }
  const std::string& dst_bn() const { return dst_bn_; }

  // Setters
  void set_lbn(const std::string& lbn) { lbn_ = lbn; }
  std::string& mut_src_bn() { return src_bn_; }
  std::string& mut_dst_bn() { return dst_bn_; }

 private:
  // various names for one blob
  std::string lbn_;
  std::string src_bn_;
  std::string dst_bn_;
};

class ExecNode final : public Node<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecNode);
  ExecNode() = default;
  ~ExecNode() = default;

  std::shared_ptr<const Operator> op() const { return op_; }
  std::shared_ptr<const Operator>& mut_op() { return op_; }

  void BindBnInOpAndRegst(const std::string&, std::weak_ptr<RegstDesc>);

  std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOpFunc() const;

  std::string VisualStr() const override { return op_->op_name(); }
  void ToProto(const ParallelContext*, ExecNodeProto*) const;

 private:
  BlobDesc* GetBlobDesc4BnInOp(const std::string&) const;

  std::shared_ptr<const Operator> op_;
  HashMap<std::string, std::weak_ptr<RegstDesc>> bn_in_op2regst_;
};

class ExecGraph final : public Graph<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecGraph);
  ExecGraph() = default;
  ~ExecGraph() = default;

  void ToExecSequence(const ParallelContext*, ExecSequence*) const;
  const char* TypeName() const override { return "ExecGraph"; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_EXEC_GRAPH_H_
