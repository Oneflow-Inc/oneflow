#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_ring_boxing_kernel_util.h"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
class CudaRingAllReduceKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRingAllReduceKernel);
  CudaRingAllReduceKernel() = default;
  ~CudaRingAllReduceKernel() override = default;

 private:
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
void CudaRingAllReduceKernel<T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto other_ctx = *static_cast<std::pair<int64_t, int64_t>*>(ctx.other);
  const int64_t step_id = other_ctx.first;
  const int64_t slice_id = other_ctx.second;
  const CudaRingAllReduceKernelConf& conf = kernel_conf().cuda_ring_all_reduce_conf();
  CudaRingBoxingStepParams<T> params{};
  const CudaRingAllReduceStepConf& step_conf = conf.step_conf(step_id);
  params.recv = step_conf.recv();
  params.in = step_conf.in();
  params.send = step_conf.send();
  params.out = step_conf.out();
  const int64_t num_link = conf.num_link();
  CHECK_GT(num_link, 0);
  CHECK_LE(num_link, kCudaRingBoxingMaxNumLink);
  params.num_links = num_link;
  Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const auto DptrWithOffsetOrNull = [](Blob* blob, int64_t offset) -> T* {
    if (blob != nullptr) {
      return blob->mut_dptr<T>() + offset;
    } else {
      return nullptr;
    }
  };
  FOR_RANGE(int64_t, link_id, 0, num_link) {
    CudaRingBoxingLinkParams<T>& link_params = params.links[link_id];
    const Range range(step_conf.link_conf(link_id).slice_range(slice_id));
    link_params.num_elem = range.size();
    link_params.send = DptrWithOffsetOrNull(BnInOp2Blob(GenRepeatedBn("send", link_id)), 0);
    link_params.recv = DptrWithOffsetOrNull(BnInOp2Blob(GenRepeatedBn("recv", link_id)), 0);
    link_params.in = DptrWithOffsetOrNull(in, range.begin());
    link_params.out = DptrWithOffsetOrNull(out, range.begin());
  }
  CudaRingBoxingKernelUtil<ReduceMethod::kSum, T>::LaunchGenericRingStep(ctx.device_ctx, params);
}

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCudaRingAllReduceConf, DeviceType::kGPU, float,
                                      CudaRingAllReduceKernel<float>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCudaRingAllReduceConf, DeviceType::kGPU,
                                      double, CudaRingAllReduceKernel<double>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCudaRingAllReduceConf, DeviceType::kGPU,
                                      float16, CudaRingAllReduceKernel<float16>);

#endif

}  // namespace oneflow
