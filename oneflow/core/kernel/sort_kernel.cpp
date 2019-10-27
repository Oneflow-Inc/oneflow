#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
void GpuSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
             std::string dir, void* temp_storage_ptr, size_t temp_storage_bytes, T* out_ptr);

template<typename T>
void CpuSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
             std::string dir, T* out_ptr) {
  Memcpy<DeviceType::kCPU>(ctx, out_ptr, in_ptr, instance_num * instance_size * sizeof(T));
  FOR_RANGE(int32_t, i, 0, instance_num) {
    T* out_ptr_i = out_ptr + i * instance_size;
    if (dir == "ASCENDING") {
      std::sort(out_ptr_i, out_ptr_i + instance_size, std::less<T>());
    } else if (dir == "DESCENDING") {
      std::sort(out_ptr_i, out_ptr_i + instance_size, std::greater<T>());
    } else {
      UNIMPLEMENTED();
    }
  }
}

template<DeviceType device_type, typename T>
class SortKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SortKernel);
  SortKernel() = default;
  ~SortKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    BnInOp2Blob("out")->dense_shape_mut_view()->set_shape(BnInOp2Blob("in")->shape());
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    int32_t instance_size = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
    int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
    const T* in_ptr = in_blob->dptr<T>();
    T* out_ptr = out_blob->mut_dptr<T>();
    if (this->op_conf().device_type() == DeviceType::kCPU) {
      CpuSort(ctx.device_ctx, in_ptr, instance_num, instance_size,
              this->op_conf().sort_conf().dir(), out_ptr);
    } else if (this->op_conf().device_type() == DeviceType::kGPU) {
      GpuSort(ctx.device_ctx, in_ptr, instance_num, instance_size,
              this->op_conf().sort_conf().dir(), BnInOp2Blob("temp_storage")->mut_dptr<void>(),
              this->kernel_conf().sort_conf().temp_storage_bytes(), out_ptr);
    } else {
      UNIMPLEMENTED();
    }
  }
};

#define REGISTER_SORT_KERNEL(dtype)                                                       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSortConf, DeviceType::kCPU, dtype, \
                                        SortKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSortConf, DeviceType::kGPU, dtype, \
                                        SortKernel<DeviceType::kGPU, dtype>)

REGISTER_SORT_KERNEL(float);
REGISTER_SORT_KERNEL(double);
REGISTER_SORT_KERNEL(int8_t);
REGISTER_SORT_KERNEL(int32_t);
REGISTER_SORT_KERNEL(int64_t);

}  // namespace oneflow
