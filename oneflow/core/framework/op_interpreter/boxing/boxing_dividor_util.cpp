#include "oneflow/core/framework/op_interpreter/boxing/boxing_dividor_util.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace private_details {

Maybe<BoxingDividor> ReplaceInDeviceType(DeviceType device_type) {
  return std::make_shared<BoxingDividor>("ReplaceInDeviceType",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        const auto& new_placement = JUST(ReplaceDeviceType(in->placement(), device_type));
        return SymbolOf(PlacedNdSbp(in->nd_sbp(), new_placement));
      });
}

Maybe<BoxingDividor> ReplaceOutDeviceType(DeviceType device_type) {
  return std::make_shared<BoxingDividor>("ReplaceOutDeviceType",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        const auto& new_placement = JUST(ReplaceDeviceType(out->placement(), device_type));
        return SymbolOf(PlacedNdSbp(out->nd_sbp(), new_placement));
      });
}

}

decltype(ReplaceInDeviceType) ReplaceInDeviceType = DECORATE(&private_details::ReplaceInDeviceType, ThreadLocal);
decltype(ReplaceOutDeviceType) ReplaceOutDeviceType = DECORATE(&private_details::ReplaceOutDeviceType, ThreadLocal);

}
