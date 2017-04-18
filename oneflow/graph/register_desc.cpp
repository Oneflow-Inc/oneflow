#include "graph/register_desc.h"
#include "job/id_manager.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  regst_desc_id_ = IDMgr::Singleton().NewRegstDescId();
  producer_ = nullptr;
}

void RegstDesc::CopyLbn2ShapeMap(const RegstDesc* rhs) {
  for (const auto& pair : rhs->lbn2shape_) {
    const std::string& lbn = pair.first;
    std::unique_ptr<Shape> shape(new Shape);
    *shape = *(pair.second);
    CHECK(lbn2shape_.insert(std::make_pair(lbn, std::move(shape))).second);
  }
}

Shape* RegstDesc::EnrollLbn(const std::string& lbn) {
  Shape* raw_ptr = new Shape;
  std::unique_ptr<Shape> uptr(raw_ptr);
  CHECK(lbn2shape_.insert(std::make_pair(lbn, std::move(uptr))).second);
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
