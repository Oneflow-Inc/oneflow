#include <cstring>
#include "oneflow/core/vm/mirrored_object.h"

namespace oneflow {

using LogicalObjIdFlatMsg = FLAT_MSG_TYPE(LogicalObjectId);

bool LogicalObjIdFlatMsg::operator<(const LogicalObjIdFlatMsg& rhs) const {
  return std::memcmp(static_cast<const void*>(this), static_cast<const void*>(&rhs),
                     sizeof(self_type))
         < 0;
}

bool LogicalObjIdFlatMsg::operator==(const LogicalObjIdFlatMsg& rhs) const {
  return std::memcmp(static_cast<const void*>(this), static_cast<const void*>(&rhs),
                     sizeof(self_type))
         == 0;
}

}  // namespace oneflow
