#include "graph/register_desc.h"
#include "job/id_manager.h"

namespace oneflow {

RegiDesc::RegiDesc() {
  regi_desc_id_ = IDMgr::Singleton().NewRegiDescId();
  producer_ = nullptr;
}

const char* ContigRegiDesc::kAllLbn = "OfReservedAllLbn";

Shape ContigRegiDesc::ComputeShape4AllLbn() const {
  TODO();
}

} // namespace oneflow
