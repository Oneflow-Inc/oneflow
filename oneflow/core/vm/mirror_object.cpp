#include <cstring>
#include "oneflow/core/vm/mirror_object.h"

namespace oneflow {

bool FLAT_MSG_TYPE(LogicalObjectPtrValue)::operator<(const FLAT_MSG_TYPE(LogicalObjectPtrValue)
                                                     & rhs) const {
  return std::memcmp(static_cast<const void*>(this), static_cast<const void*>(&rhs),
                     sizeof(self_type))
         < 0;
}

bool FLAT_MSG_TYPE(LogicalObjectPtrValue)::operator==(const FLAT_MSG_TYPE(LogicalObjectPtrValue)
                                                      & rhs) const {
  return std::memcmp(static_cast<const void*>(this), static_cast<const void*>(&rhs),
                     sizeof(self_type))
         == 0;
}

}  // namespace oneflow
