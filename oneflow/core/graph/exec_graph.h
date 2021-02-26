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
  ExecNode() {}
  ~ExecNode() = default;

  std::shared_ptr<const Operator> op() const { return op_; }
  std::shared_ptr<const Operator>& mut_op() { return op_; }
  RegstDesc* RegstDesc4BnInOp(const std::string& bn) const { return bn_in_op2regst_.at(bn).get(); }

  void BindBnWithRegst(const std::string& bn, std::shared_ptr<RegstDesc>);
  void BindBnsWithRegst(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                        std::shared_ptr<RegstDesc>);
  void AddBnToRegstAndBindIt(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                             std::shared_ptr<RegstDesc>);
  bool TryBindBnWithOneOfTheRegsts(const std::string&,
                                   const std::list<std::shared_ptr<RegstDesc>>&);
  void BindBnWithOneOfTheRegsts(const std::string&, const std::list<std::shared_ptr<RegstDesc>>&);
  void UnbindBnWithEmptyRegst();

  std::string VisualStr() const override { return op_->op_name(); }
  void ToProto(const ParallelContext*, ExecNodeProto*) const;

  void InferBlobDescs(const ParallelContext* parallel_ctx);

  const HashMap<std::string, std::string>& mut_inplace_obn2ibn() const {
    return mut_inplace_obn2ibn_;
  }
  const HashMap<std::string, std::string>& con_inplace_obn2ibn() const {
    return con_inplace_obn2ibn_;
  }

 private:
  std::function<const BlobDesc&(const std::string&)> GetLogicalBlobDesc4BnInOpFunc() const;
  std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOpFunc() const;

  std::shared_ptr<const Operator> op_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> bn_in_op2regst_;

  HashMap<std::string, std::string> mut_inplace_obn2ibn_;
  HashMap<std::string, std::string> con_inplace_obn2ibn_;
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
