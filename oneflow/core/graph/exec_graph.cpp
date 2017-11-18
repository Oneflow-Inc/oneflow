#include "oneflow/core/graph/exec_graph.h"

namespace oneflow {

void ExecNode::BindBnInOpAndRegst(const std::string& bn_in_op,
                                  std::weak_ptr<RegstDesc> regst) {
  CHECK(bn_in_op2regst_.emplace(bn_in_op, regst).second);
}

std::function<BlobDesc*(const std::string&)> ExecNode::GetBlobDesc4BnInOpFunc()
    const {
  return std::bind(&ExecNode::GetBlobDesc4BnInOp, this, std::placeholders::_1);
}

void ExecNode::ToProto(const ParallelContext* parallel_ctx,
                       TodoExecNodeProto* ret) const {
  op_->GenKernelConf(GetBlobDesc4BnInOpFunc(), parallel_ctx,
                     ret->mutable_kernel_conf());
  for (const auto& bn_regst : bn_in_op2regst_) {
    const std::string& bn_in_op = bn_regst.first;
    auto regst = bn_regst.second.lock();
    if (!regst) { continue; }
    PbMapPair<std::string, int64_t> pair{bn_in_op, regst->regst_desc_id()};
    CHECK(ret->mutable_bn_in_op2regst_desc_id()->insert(pair).second);
  }
}

BlobDesc* ExecNode::GetBlobDesc4BnInOp(const std::string& bn_in_op) const {
  auto it = bn_in_op2regst_.find(bn_in_op);
  if (it == bn_in_op2regst_.end()) { return nullptr; }
  std::shared_ptr<RegstDesc> regst = it->second.lock();
  if (!regst) { return nullptr; }
  const std::string& lbn = this->op()->Lbn4BnInOp(bn_in_op);
  return regst->MutBlobDesc(lbn);
}

void ExecGraph::ToExecSequence(const ParallelContext* parallel_ctx,
                               TodoExecSequence* ret) const {
  TopoForEachNode([&](ExecNode* node) {
    node->ToProto(parallel_ctx, ret->add_exec_node());
  });
}

}  // namespace oneflow
