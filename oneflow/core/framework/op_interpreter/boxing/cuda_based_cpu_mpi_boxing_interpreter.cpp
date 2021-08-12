#include "oneflow/core/framework/op_interpreter/boxing/cuda_based_cpu_mpi_boxing_interpreter.h"
#include "oneflow/core/functional/function.h"

namespace oneflow {

namespace {

struct CudaBaseMpiEagerBoxingCall {
  std::shared_ptr<EagerBoxingCall> opt_h2d;
  std::shared_ptr<EagerBoxingCall> cud_mpi;
  std::shared_ptr<EagerBoxingCall> opt_h2d;
};

Maybe<CudaBaseMpiEagerBoxingCall> RawGetCudaBaseMpiEagerBoxingCall(
    Symbol<cfg::ParallelDistribution> in_nd_sbp, Symbol<cfg::ParallelDistribution> out_nd_sbp,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) {
  const auto* mgr = Global<EagerBoxingInterpreterManager>::Get();
  const auto& gpu_in_parallel_desc = JUST(ReplaceDeviceType(in_parallel_desc, DeviceType::kGPU));
  const auto& gpu_out_parallel_desc = JUST(ReplaceDeviceType(out_parallel_desc, DeviceType::kGPU));
  CHECK_OR_RETURN(gpu_in_parallel_desc == gpu_out_parallel_desc);
  const auto& opt_h2d =
    JUST(EagerBoxingCall::New(in_nd_sbp, in_nd_sbp, in_parallel_desc, gpu_in_parallel_desc));
  const auto& gpu_mpi =
    JUST(EagerBoxingCall::New(in_nd_sbp, out_nd_sbp, gpu_in_parallel_desc, gpu_out_parallel_desc));
  const auto& opt_d2h = 
    JUST(EagerBoxingCall::New(out_nd_sbp, out_nd_sbp, gpu_out_parallel_desc, out_parallel_desc));
  return std::shared_ptr<CudaBaseMpiEagerBoxingCall>(new CudaBaseMpiEagerBoxingCall{
    .opt_h2d=opt_h2d, .gpu_mpi=gpu_mpi, .opt_d2h=opt_d2h,
  }); 
}

static constexpr auto* GetCudaBaseMpiEagerBoxingCall =
    DECORATE(&RawGetCudaBaseMpiEagerBoxingCall, ThreadLocal);
}

Maybe<one::Tensor> CudaBasedCpuMpiBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::ParallelDistribution> in_nd_sbp,
    Symbol<cfg::ParallelDistribution> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  const auto& call = JUST(GetCudaBaseMpiEagerBoxingCall(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  auto tensor = input;
  tensor = call->opt_h2d->Apply(tensor);
  tensor = call->gpu_mpi->Apply(tensor);
  tensor = call->opt_d2h->Apply(tensor);
  return tensor;
}

}
