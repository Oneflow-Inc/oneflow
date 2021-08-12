#include "oneflow/core/framework/op_interpreter/boxing/cuda_copy_boxing_interpreter.h"
#include "oneflow/core/functional/function.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

Maybe<one::Tensor> CudaCopyBoxingInterpreter::InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                 Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                 Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                 Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(in_nd_sbp == out_nd_sbp);
  CHECK_EQ_OR_RETURN(in_parallel_desc->device_type(), DeviceType::kCPU);
  CHECK_EQ_OR_RETURN(out_parallel_desc->device_type(), DeviceType::kGPU);
  CHECK_OR_RETURN(JUST(ReplaceDeviceTag(in_parallel_desc, DeviceType::kGPU)) == out_parallel_desc);
  const auto& local_tensor = JUST(input->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out_nd_sbp));
  return functional::ToConsistent(local_tensor, out_parallel_desc, *sbp_list, Optional<Shape>());
}

}
