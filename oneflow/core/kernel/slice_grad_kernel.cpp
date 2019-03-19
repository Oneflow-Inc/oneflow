#include "oneflow/core/kernel/slice_kernel.h"
#include "oneflow/core/kernel/slice_grad_kernel.h"
#include "oneflow/core/ndarray/cpu_ndarray_builder.h"

namespace oneflow {

template<typename T>
void SliceGradKernel<DeviceType::kCPU, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const SliceGradOpConf& conf = this->op_conf().slice_grad_conf();
  const Blob* dy_blob = BnInOp2Blob("dy");
  Blob* dx_blob = BnInOp2Blob("dx");
  CHECK_EQ(dy_blob->shape().NumAxes(), dx_blob->shape().NumAxes());

  Memset<DeviceType::kCPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                           dx_blob->ByteSizeOfDataContentField());

  switch (dx_blob->shape().NumAxes()) {
// clang-format off
#define MAKE_CASE(num_axes)                                                           \
    case num_axes: {                                                                  \
      NdarraySliceUtil<T, num_axes>::Backward(ctx.device_ctx, conf.dim_slice_conf(),  \
                                              dy_blob, dx_blob);           \
      break;                                                                          \
    }
    MAKE_CASE(2);
    MAKE_CASE(3);
#undef MAKE_CASE
    // clang-format on
    default: { UNIMPLEMENTED(); }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceConf, SliceGradKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
