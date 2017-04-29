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

Shape* RegstDesc::EnrollLbn(const std::string& lbn) {
  Shape* raw_ptr = new Shape;
  std::unique_ptr<Shape> uptr(raw_ptr);
  CHECK(lbn2shape_.emplace(lbn, std::move(uptr)).second);
  return raw_ptr;
}

const Shape& RegstDesc::GetShape(const std::string& lbn) {
  return *(lbn2shape_.at(lbn));
}

Shape* RegstDesc::GetMutShapePtr(const std::string& lbn) {
  return lbn2shape_.at(lbn).get();
}

const char* RegstDesc::kAllLbn = "OfReservedAllLbn";

} // namespace oneflow
