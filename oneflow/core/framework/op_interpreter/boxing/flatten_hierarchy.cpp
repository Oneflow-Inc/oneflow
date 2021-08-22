#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Maybe<void> RawCheckFlattenHierarchy(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_GT_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  for (int i = 0; i < in->nd_sbp()->sbp_parallel(); ++i) {
    const auto& sbp_parallel = in->nd_sbp()->sbp_parallel(i);
    CHECK_OR_RETURN(sbp_parallel == out->nd_sbp()->sbp_parallel(0))
      << "nd_sbp axis: " << i;
  }
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), out->placement()->device_type());
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), out->placement()->parallel_num());
  ParallelConf flattened_parallel_conf(in->placement()->parallel_conf());
  flattened_parallel_conf.clear_hierarchy();
  const auto& flatten_placement = SymbolOf(ParallelDesc(flattened_parallel_conf));
  CHECK_OR_RETURN(flatten_placement == out->placement())
    << "The output placement is not a hierarch-flattened version of the input placement";
  return Maybe<void>::Ok();
}

}

static constexpr auto* CheckFlattenHierarchy = DECORATE(&RawCheckFlattenHierarchy, ThreadLocal);

Maybe<one::Tensor> FlattenHierarchy(
    const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_OR_RETURN(JUST(tenosr->nd_sbp()) == in->nd_sbp());
  CHECK_OR_RETURN(JUST(tenosr->parallel_desc()) == in->placement());
  const auto& local_tensor = JUST(tensor->cur_rank_phy_tensor());
  return JUST(one::functional::LocalToConsistent(
        tensor, out->placement(), out->nd_sbp(), tensor->shape(), tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("flatten-hierarchy", &CheckFlattenHierarchy, &FlattenHierarchy));

}
