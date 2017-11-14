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

void ExecNode::ToProto(ExecNodeProto* ret) const {
  ret->set_op_name(op_->op_name());
  for (const auto& bn_regst : bn_in_op2regst_) {
    auto regst = bn_regst.second.lock();
    if (regst) {
      ret->mutable_bn_in_op2regst_desc_id()->insert(
          {bn_regst.first, regst->regst_desc_id()});
    }
  }
}

BlobDesc* ExecNode::GetBlobDesc4BnInOp(const std::string& bn_in_op) const {
  auto it = this->bn_in_op2regst_.find(bn_in_op);
  if (it == this->bn_in_op2regst_.end()) { return nullptr; }
  std::shared_ptr<RegstDesc> regst = it->second.lock();
  const std::string& lbn = this->op()->Lbn4BnInOp(bn_in_op);
  return regst->MutBlobDesc(lbn);
}

void ExecGraph::ToExecSequence(ExecSequence* ret) const {
  TopoForEachNode([&](ExecNode* node) { node->ToProto(ret->add_exec_node()); });
}

}  // namespace oneflow
