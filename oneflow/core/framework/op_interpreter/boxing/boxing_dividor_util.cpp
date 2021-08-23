#include "oneflow/core/framework/op_interpreter/boxing/boxing_dividor_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

Maybe<BoxingDividor> RawReplaceInDeviceType(DeviceType device_type) {
  return std::make_shared<BoxingDividor>("ReplaceInDeviceType",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        const auto& new_placement = JUST(ReplaceDeviceType(in->placement(), device_type));
        return PlacedNdSbp::New(in->nd_sbp(), new_placement);
      });
}

Maybe<BoxingDividor> RawReplaceOutDeviceType(DeviceType device_type) {
  return std::make_shared<BoxingDividor>("ReplaceOutDeviceType",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        const auto& new_placement = JUST(ReplaceDeviceType(out->placement(), device_type));
        return PlacedNdSbp::New(out->nd_sbp(), new_placement);
      });
}

}

decltype(ReplaceInDeviceType) ReplaceInDeviceType = DECORATE(&RawReplaceInDeviceType, ThreadLocal);
decltype(ReplaceOutDeviceType) ReplaceOutDeviceType = DECORATE(&RawReplaceOutDeviceType, ThreadLocal);

namespace {

Maybe<Symbol<PlacedNdSbp>> RawFlattenHierarchy(Symbol<PlacedNdSbp> placed_nd_sbp) {
  CHECK_GE_OR_RETURN(placed_nd_sbp->nd_sbp()->sbp_parallel_size(), 0);
  const auto& first_sbp_parallel = placed_nd_sbp->nd_sbp()->sbp_parallel(0);
  for (const auto& sbp_parallel : placed_nd_sbp->nd_sbp()->sbp_parallel()) {
    CHECK_OR_RETURN(sbp_parallel == first_sbp_parallel);
  }
  std::vector<Symbol<cfg::SbpParallel>> vec{Symbol(first_sbp_parallel)};
  const auto& flattened_nd_sbp = GetNdSbp(vec);
  ParallelConf flattened_parallel_conf(in->placement()->parallel_conf());
  flattened_parallel_conf.clear_hierarchy();
  const auto& flattened_placement = SymbolOf(ParallelDesc(flattened_parallel_conf));
  return JUST(PlacedNdSbp::New(flattened_nd_sbp, flattened_placement)); 
}

static constexpr auto* FlattenHierarchy = DECORATE(&RawFlattenHierarchy, ThreadLocal);

Maybe<BoxingDividor> RawFlattenInHierarchy() {
  return std::make_shared<BoxingDividor>("FlattenInHierarchy",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return FlattenHierarchy(in);
      });
}

}

decltype(FlattenInHierarchy) FlattenInHierarchy = DECORATE(&RawFlattenInHierarchy, ThreadLocal);
}
