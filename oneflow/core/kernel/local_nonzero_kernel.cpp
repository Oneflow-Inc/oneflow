#include "oneflow/core/kernel/local_nonzero_kernel.h"

namespace oneflow {

namespace {

void SetIndex(int64_t offset, const Shape& shape, int32_t* index_ptr) {
  int64_t dim_elem_cnt = shape.elem_cnt();
  for (int64_t i = 0; i < shape.NumAxes(); ++i) {
    dim_elem_cnt /= shape.At(i);
    index_ptr[i] = static_cast<int32_t>(offset / dim_elem_cnt);
    offset %= dim_elem_cnt;
  }
}

}  // namespace

template<DeviceType device_type, typename T>
void LocalNonzeroKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  if (this->op_conf().device_type() == DeviceType::kGPU) {
    Blob* num_nonzero_blob = BnInOp2Blob("num_nonzero");
    Blob* shape_blob = BnInOp2Blob("shape");
    LocalNonzeroUtil<T>::ForwardGpu(ctx.device_ctx, in_blob, num_nonzero_blob, shape_blob,
                                    out_blob);
  } else if (this->op_conf().device_type() == DeviceType::kCPU) {
    LocalNonzeroUtil<T>::ForwardCpu(ctx.device_ctx, in_blob, out_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void LocalNonzeroKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // already done in ForwardDataContent
}

template<typename T>
void LocalNonzeroUtil<T>::ForwardCpu(DeviceCtx* ctx, const Blob* in_blob, Blob* out_blob) {
  int64_t num_nonzero = 0;
  const Shape shape = in_blob->shape();
  FOR_RANGE(int64_t, i, 0, in_blob->shape().elem_cnt()) {
    if (in_blob->dptr<T>()[i] != ZeroVal<T>::value) {
      SetIndex(i, shape, out_blob->mut_dptr<int32_t>() + num_nonzero * shape.NumAxes());
      num_nonzero += 1;
    }
  }
  out_blob->set_dim0_valid_num(0, num_nonzero);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalNonzeroConf, LocalNonzeroKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
