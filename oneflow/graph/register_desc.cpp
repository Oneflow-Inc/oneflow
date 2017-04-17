#include "graph/register_desc.h"
#include "job/id_manager.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  regst_desc_id_ = IDMgr::Singleton().NewRegstDescId();
  producer_ = nullptr;
}

const char* ContigRegstDesc::kAllLbn = "OfReservedAllLbn";

} // namespace oneflow
