#include "register/register_manager.h"

namespace oneflow {

void NewRegsts(const RegstDescProto& regst_desc_proto,
               std::function<void(Regst*)> OneRegstDone) {
  // One RegstDesc means Multi Regst
  // All Regst has a shared_ptr point to the same RtRegstDesc obj
  // Call OneRegstDone for each regst
}

}
