#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
}

void ExecNode::UnBindRegstsWithZeroBlobSize() {
  EraseIf<std::string, std::weak_ptr<RegstDesc>>(&bn_in_op2regst_, [this]
      (HashMap<std::string, std::weak_ptr<RegstDesc>>::iterator it) {
    const std::string& lbn = this->op_->Lbn4BnInOp(it->first);
    if (lbn == RegstDesc::kAllLbn) {
      return it->second.lock()->CompElemCntOfAllBlob() == 0;
    } else {
      return it->second.lock()->GetShape(lbn).elem_cnt() == 0;
    }
  });
}

std::function<Shape*(const std::string&)>
ExecNode::GetMutShapePtr4BnInOpFunc() const {
  return [this](const std::string& bn_in_op) -> Shape* {
    auto it = this->bn_in_op2regst_.find(bn_in_op);
    if (it == this->bn_in_op2regst_.end()) { return nullptr; }
    std::shared_ptr<RegstDesc> regst = it->second.lock();
    const std::string& lbn = this->op()->Lbn4BnInOp(bn_in_op);
    return regst->GetMutShapePtr(lbn);
  };
}

void ExecNode::ToProto(ExecNodeProto* ret) const {
  ret->set_op_name(op_->op_name());
  for (const auto& bn_regst: bn_in_op2regst_) {
    ret->mutable_bn_in_op2regst_desc_id()->insert({
        bn_regst.first, bn_regst.second.lock()->regst_desc_id()});
  }
}

void ExecGraph::ToExecSequence(ExecSequence* ret) const {
  for (const ExecNode& node: *this) {
    if (!node.bn_in_op2regst().empty()) {
      node.ToProto(ret->add_exec_node());
    }
  }
}

} // namespace oneflow
