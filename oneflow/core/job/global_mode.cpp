
#include "oneflow/core/job/global_mode.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

/* static */ bool* GlobalMode::get_mode_ptr() {
  static thread_local bool mode = false;
  return &mode;
}
/* static */ bool GlobalMode::is_enabled() { return *get_mode_ptr(); }
/* static */ void GlobalMode::set_enabled(bool enabled) { *get_mode_ptr() = enabled; }

/* static */ Symbol<NdSbp>* GlobalMode::get_nd_sbp_ptr() {
  static thread_local Symbol<NdSbp> nd_sbp;
  return &nd_sbp;
}
/* static */ Symbol<NdSbp> GlobalMode::nd_sbp() { return *get_nd_sbp_ptr(); }
/* static */ void GlobalMode::set_nd_sbp(Symbol<NdSbp> nd_sbp) { *get_nd_sbp_ptr() = nd_sbp; }

/* static */ Symbol<ParallelDesc>* GlobalMode::get_parallel_desc_ptr() {
  static thread_local Symbol<ParallelDesc> parallel_desc;
  return &parallel_desc;
}
/* static */ Symbol<ParallelDesc> GlobalMode::parallel_desc() { return *get_parallel_desc_ptr(); }
/* static */ void GlobalMode::set_parallel_desc(Symbol<ParallelDesc> parallel_desc) { *get_parallel_desc_ptr() = parallel_desc; }


}

