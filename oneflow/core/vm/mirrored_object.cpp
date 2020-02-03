#include <cstring>
#include "oneflow/core/vm/mirrored_object.h"

namespace oneflow {

using LogicalObjPtrFlatMsg = FLAT_MSG_TYPE(LogicalObjectPtrValue);

bool LogicalObjPtrFlatMsg::operator<(const LogicalObjPtrFlatMsg& rhs) const {
  return std::memcmp(static_cast<const void*>(this), static_cast<const void*>(&rhs),
                     sizeof(self_type))
         < 0;
}

bool LogicalObjPtrFlatMsg::operator==(const LogicalObjPtrFlatMsg& rhs) const {
  return std::memcmp(static_cast<const void*>(this), static_cast<const void*>(&rhs),
                     sizeof(self_type))
         == 0;
}

}  // namespace oneflow
