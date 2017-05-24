#include "register/register_desc.h"
#include "common/id_manager.h"
#include "common/protobuf.h"
#include "graph/task_node.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  producer_ = nullptr;
  register_num_ = 5; // TODO
}

void RegstDesc::AddSubscriber(const TaskNode* new_subscriber) {
  CHECK(subscribers_.insert(new_subscriber).second);
}

void RegstDesc::CopyLbnFrom(const RegstDesc* rhs) {
  lbn2shape_.clear();
  for (const auto& pair : rhs->lbn2shape_) {
    const std::string& lbn = pair.first;
    auto shape = of_make_unique<Shape> ();
    CHECK(lbn2shape_.emplace(lbn, std::move(shape)).second);
  }
}

void RegstDesc::CopyShapeFrom(const RegstDesc* rhs) {
  for (const auto& pair : lbn2shape_) {
    const std::string& lbn = pair.first;
    *(lbn2shape_.at(lbn)) = rhs->GetShape(lbn);
  }
}

void RegstDesc::EnrollLbn(const std::string& lbn) {
  std::unique_ptr<Shape> ptr(new Shape);
  CHECK(lbn2shape_.emplace(lbn, std::move(ptr)).second) << lbn;
}

const Shape& RegstDesc::GetShape(const std::string& lbn) const {
  return *(lbn2shape_.at(lbn));
}

Shape* RegstDesc::GetMutShapePtr(const std::string& lbn) {
  return lbn2shape_.at(lbn).get();
}

HashMap<std::string, std::unique_ptr<Shape>>& RegstDesc::mut_lbn2shape() {
  return lbn2shape_;
}

const HashMap<std::string, std::unique_ptr<Shape>>&
RegstDesc::lbn2shape() const {
  return lbn2shape_;
}

void RegstDesc::EraseZeroSizeBlob() {
  EraseIf<std::string, std::unique_ptr<Shape>>(&lbn2shape_, []
      (HashMap<std::string, std::unique_ptr<Shape>>::iterator it) {
    return it->second->elem_cnt() == 0;
  });
}

int64_t RegstDesc::CompElemCntOfAllBlob() const {
  int64_t sum = 0;
  for (const auto& pair : lbn2shape_) {
    sum += pair.second->elem_cnt();
  }
  return sum;
}

std::string RegstDesc::DebugStr() const {
  std::stringstream ss;
  ss << "{";
  for (const auto& pair : lbn2shape_) {
    ss << "{" << pair.first << ":" << pair.second->DebugStr() << "}"; 
  }
  ss << "}";
  return ss.str();
}

void RegstDesc::ToProto(RegstDescProto* ret) const {
  ret->set_regst_desc_id(regst_desc_id_);
  ret->set_producer_task_id(producer_->task_id());
  for (const TaskNode* subscriber : subscribers_) {
    ret->add_subscriber_task_id(subscriber->task_id());
  }
  for (const auto& pair : lbn2shape_) {
    PbMapPair<std::string, ShapeProto> pb_pair(pair.first);
    pair.second->ToProto(&(pb_pair.second));
    ret->mutable_lbn2shape()->insert(pb_pair);
  }
  ret->set_register_num(register_num_);
  *(ret->mutable_mem_case()) = InferMemCase();
}

MemoryCase RegstDesc::InferMemCase() const {
  // TODO
  return MemoryCase();
}

const char* RegstDesc::kAllLbn = "OfReservedAllLbn";

} // namespace oneflow
