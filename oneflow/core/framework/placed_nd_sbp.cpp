#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Maybe<Symbol<PlacedNdSbp>> RawNew(const Symbol<cfg::NdSbp>& nd_sbp, const Symbol<ParallelDesc>& placement) {
  CHECK_OR_RETURN(!nd_sbp);
  CHECK_OR_RETURN(!placement);
  CHECK_GT_OR_RETURN(nd_sbp->sbp_parallel_size(), 0);
  CHECK_EQ_OR_RETURN(nd_sbp->sbp_parallel_size(), placement->hierarchy()->NumAxes());
  return SymbolOf(PlacedNdSbp(nd_sbp, placement));
}

}

decltype(PlacedNdSbp::New) PlacedNdSbp::New = DECORATE(&RawNew, ThreadLocal);

}
