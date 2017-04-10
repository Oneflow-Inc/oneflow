#include "graph/register_desc.h"
#include "job/id_manager.h"

namespace oneflow {

RegisterDesc::RegisterDesc() {
  register_desc_id_ = IDMgr::Singleton().NewRegisterDescId();
  producer_ = nullptr;
}

const char* ContigRegistDesc::kLogicalAllBlobName = "OfReservedLogicalAllBlobName";

Shape ContigRegistDesc::ComputeShape4AllLbn() const {
  TODO();
}

} // namespace oneflow
