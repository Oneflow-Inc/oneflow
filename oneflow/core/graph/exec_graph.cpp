#include "oneflow/core/graph/exec_graph.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void ExecNode::BindBnWithRegst(const std::string& bn, std::shared_ptr<RegstDesc> regst) {
  CHECK(bn_in_op2regst_.emplace(bn, regst).second);
}

void ExecNode::BindBnsWithRegst(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                                std::shared_ptr<RegstDesc> regst) {
  for (const std::string& bn : (op_.get()->*bns_getter)()) { BindBnWithRegst(bn, regst); }
}

void ExecNode::AddBnToRegstAndBindIt(const PbRpf<std::string>& (Operator::*bns_getter)() const,
                                     std::shared_ptr<RegstDesc> regst) {
  for (const std::string& bn : (op_.get()->*bns_getter)()) { regst->AddLbi(op_->BnInOp2Lbi(bn)); }
  BindBnsWithRegst(bns_getter, regst);
}

void ExecNode::BindBnWithOneOfTheRegsts(const std::string& bn,
                                        const std::list<std::shared_ptr<RegstDesc>>& regsts) {
  const LogicalBlobId& lbi = op()->BnInOp2Lbi(bn);
  bool has_binded = false;
  for (std::shared_ptr<RegstDesc> regst : regsts) {
    if (regst->GetBlobDesc(lbi) == nullptr) { continue; }
    BindBnWithRegst(bn, regst);
    has_binded = true;
    break;
  }
  CHECK(has_binded);
}

void ExecNode::UnbindBnWithEmptyRegst() {
  EraseIf<std::string, std::shared_ptr<RegstDesc>>(
      &bn_in_op2regst_, [](HashMap<std::string, std::shared_ptr<RegstDesc>>::iterator it) {
        return it->second->regst_desc_type().has_data_regst_desc() && it->second->NumOfLbi() == 0;
      });
}

void ExecNode::ToProto(bool is_forward, const ParallelContext* parallel_ctx,
                       ExecNodeProto* ret) const {
  op_->GenKernelConf(GetBlobDesc4BnInOpFunc(), is_forward, parallel_ctx, ret->mutable_kernel_conf(),
                     op_context());
  for (const auto& bn_regst : bn_in_op2regst_) {
    const std::string& bn_in_op = bn_regst.first;
    auto regst = bn_regst.second;
    CHECK(regst);
    PbMapPair<std::string, int64_t> pair{bn_in_op, regst->regst_desc_id()};
    CHECK(ret->mutable_bn_in_op2regst_desc_id()->insert(pair).second);
  }
}

void ExecNode::InferBlobDescs(const ParallelContext* parallel_ctx) {
  auto GetBlobDesc4BnInOp = GetBlobDesc4BnInOpFunc();
  op_->InferBlobDescsIf(GetBlobDesc4BnInOp, parallel_ctx, Global<JobDesc>::Get()->RecordPieceSize(),
                        [this](OpContext* op_ctx) { op_ctx_.reset(op_ctx); });
  Global<OpGraph>::Get()->CheckBlobDescs(op_->op_name(), GetBlobDesc4BnInOp, parallel_ctx);
}

void ExecNode::InferBwBufBlobDescs(const ParallelContext* parallel_ctx) {
  op_->InferBwBufBlobDescsIf(GetBlobDesc4BnInOpFunc(), parallel_ctx, op_context());
}

void ExecNode::InferDiffBlobDescsWithoutFwNode(const ParallelContext* parallel_ctx) {
  op_->InferDiffBlobDescsWithoutFwBlob(GetBlobDesc4BnInOpFunc(), parallel_ctx);
}
void ExecNode::FixInDiffBlobDescs(const ParallelContext* parallel_ctx) {
  op_->FixInDiffBlobDescs(GetBlobDesc4BnInOpFunc(), parallel_ctx);
}

std::function<BlobDesc*(const std::string&)> ExecNode::GetBlobDesc4BnInOpFunc() const {
  return [this](const std::string& bn_in_op) -> BlobDesc* {
    auto it = bn_in_op2regst_.find(bn_in_op);
    if (it == bn_in_op2regst_.end()) { return nullptr; }
    std::shared_ptr<RegstDesc> regst = it->second;
    CHECK(regst);
    return regst->MutBlobDesc(op()->BnInOp2Lbi(bn_in_op));
  };
}

void ExecGraph::ToExecSequence(bool is_forward, const ParallelContext* parallel_ctx,
                               ExecSequence* ret) const {
  TopoForEachNode(
      [&](ExecNode* node) { node->ToProto(is_forward, parallel_ctx, ret->add_exec_node()); });
}

}  // namespace oneflow
