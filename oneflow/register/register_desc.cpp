#include "register/register_desc.h"
#include "job/id_manager.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  producer_ = nullptr;
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

std::string RegstDesc::DebugStr() const {
  std::stringstream ss;
  ss << "{";
  for (const auto& pair : lbn2shape_) {
    ss << "{" << pair.first << ":" << pair.second->DebugStr() << "}"; 
  }
  ss << "}";
  return ss.str();
}

const char* RegstDesc::kAllLbn = "OfReservedAllLbn";

} // namespace oneflow
