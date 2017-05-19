#include "register/register.h"

namespace oneflow {

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

}
