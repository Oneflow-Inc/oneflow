#include "oneflow/core/kernel/kernel.h"

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

template<typename T>
void GpuNonzero(DeviceCtx* ctx, const Blob* in_blob, Blob* shape_blob, Blob* out_blob,
                Blob* num_nonzero_blob);

template<typename T>
void CpuNonzero(DeviceCtx* ctx, const Blob* in_blob, Blob* out_blob, Blob* num_nonzero_blob) {
  int64_t num_nonzero = 0;
  const Shape shape = in_blob->shape();
  FOR_RANGE(int64_t, i, 0, shape.elem_cnt()) {
    if (in_blob->dptr<T>()[i] != GetZeroVal<T>()) {
      SetIndex(i, shape, out_blob->mut_dptr<int32_t>() + num_nonzero * shape.NumAxes());
      num_nonzero += 1;
    }
  }
}

template<DeviceType device_type, typename T>
class NonzeroKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonzeroKernel);
  NonzeroKernel() = default;
  ~NonzeroKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    Blob* num_nonzero_blob = BnInOp2Blob("num_nonzero");

    if (this->op_conf().device_type() == DeviceType::kGPU) {
      Blob* shape_blob = BnInOp2Blob("shape");
      GpuNonzero<T>(ctx.device_ctx, in_blob, shape_blob, out_blob, num_nonzero_blob);
    } else if (this->op_conf().device_type() == DeviceType::kCPU) {
      CpuNonzero<T>(ctx.device_ctx, in_blob, out_blob, num_nonzero_blob);
    } else {
      UNIMPLEMENTED();
    }
  }
};

#define REGISTER_NONZERO_KERNEL(dtype)                                                            \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalNonzeroConf, DeviceType::kCPU, dtype, \
                                        NonzeroKernel<DeviceType::kCPU, dtype>)                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalNonzeroConf, DeviceType::kGPU, dtype, \
                                        NonzeroKernel<DeviceType::kGPU, dtype>)

REGISTER_NONZERO_KERNEL(float);
REGISTER_NONZERO_KERNEL(double);
REGISTER_NONZERO_KERNEL(int8_t);
REGISTER_NONZERO_KERNEL(int32_t);
REGISTER_NONZERO_KERNEL(int64_t);

}  // namespace oneflow
