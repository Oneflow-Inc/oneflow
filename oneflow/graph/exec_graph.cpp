#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
}

void ExecNode::UnBindRegstsWithZeroBlobSize() {
  EraseIf<std::string, RegstDesc*>(&bn_in_op2regst_, [this]
      (HashMap<std::string, RegstDesc*>::iterator it) {
    const std::string& lbn = this->op_->Lbn4BnInOp(it->first);
    if (lbn == RegstDesc::kAllLbn) {
      return it->second->CompElemCntOfAllBlob() == 0;
    } else {
      return it->second->GetShape(lbn).elem_cnt() == 0;
    }
  });
}

std::function<Shape*(const std::string&)>
ExecNode::GetMutShapePtr4BnInOpFunc() const {
  return [this](const std::string& bn_in_op) -> Shape* {
    auto it = this->bn_in_op2regst_.find(bn_in_op);
    if (it == this->bn_in_op2regst_.end()) { return nullptr; }
    RegstDesc* regst = it->second;
    const std::string& lbn = this->op()->Lbn4BnInOp(bn_in_op);
    return regst->GetMutShapePtr(lbn);
  };
}

void ExecNode::ToProto(ExecNodeProto* ret) const {
  ret->set_op_name(op_->op_name());
  for (const std::pair<std::string, RegstDesc*>& bn_regst: bn_in_op2regst_) {
    ret->mutable_bn_in_op2regst_desc_id()->insert({
        bn_regst.first, bn_regst.second->regst_desc_id()});
  }
}

RegstDesc* ExecGraph::RelatedModelRegst() const {
  for (const auto& exec_node : nodes()) {
    for (const std::string& mbn : exec_node->op()->model_bns()) {
      return exec_node->GetRegstFromBnInOp(mbn);
    }
  }
  return nullptr;
}

void ExecGraph::ToExecSequence(ExecSequence* ret) const {
  for (const ExecNode& node: *this) {
    if (!node.bn_in_op2regst().empty()) {
      node.ToProto(ret->add_exec_node());
    }
  }
}

} // namespace oneflow
