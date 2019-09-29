#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
void GpuArgSort(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
                int32_t instance_size, std::string dir, void* temp_storage_ptr,
                size_t temp_storage_bytes, T* sorted_in_ptr, int32_t* out_ptr);

template<typename T>
void CpuArgSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
                std::string dir, int32_t* out_ptr) {
  FOR_RANGE(int32_t, i, 0, instance_num) {
    const T* in_ptr_i = in_ptr + i * instance_size;
    int32_t* out_ptr_i = out_ptr + i * instance_size;
    std::iota(out_ptr_i, out_ptr_i + instance_size, 0);
    auto comp = [&](const int32_t lhs, const int32_t rhs) {
      const T l = in_ptr_i[lhs];
      const T r = in_ptr_i[rhs];
      if (l == r) {
        return lhs < rhs;
      } else {
        if (dir == "ASCENDING") {
          return l < r;
        } else if (dir == "DESCENDING") {
          return l > r;
        } else {
          UNIMPLEMENTED();
        }
      }
    };
    std::sort(out_ptr_i, out_ptr_i + instance_size, comp);
  }
}

template<DeviceType device_type, typename T>
class ArgSortKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgSortKernel);
  ArgSortKernel() = default;
  ~ArgSortKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    BnInOp2Blob("out")->dense_shape_mut_view().set_shape(
        static_cast<Shape>(BnInOp2Blob("in")->shape()));
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    int32_t instance_size = in_blob->shape().dim_vec().back();
    int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
    const T* in_ptr = in_blob->dptr<T>();
    int32_t* out_ptr = out_blob->mut_dptr<int32_t>();
    if (this->op_conf().device_type() == DeviceType::kCPU) {
      CpuArgSort(ctx.device_ctx, in_ptr, instance_num, instance_size,
                 this->op_conf().arg_sort_conf().dir(), out_ptr);
    } else if (this->op_conf().device_type() == DeviceType::kGPU) {
      GpuArgSort(ctx.device_ctx, in_ptr, BnInOp2Blob("indices")->mut_dptr<int32_t>(), instance_num,
                 instance_size, this->op_conf().arg_sort_conf().dir(),
                 BnInOp2Blob("temp_storage")->mut_dptr<void>(),
                 this->kernel_conf().arg_sort_conf().temp_storage_bytes(),
                 BnInOp2Blob("sorted_in")->mut_dptr<T>(), out_ptr);
    } else {
      UNIMPLEMENTED();
    }
  }
};

#define REGISTER_ARG_SORT_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kArgSortConf, DeviceType::kCPU, dtype, \
                                        ArgSortKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kArgSortConf, DeviceType::kGPU, dtype, \
                                        ArgSortKernel<DeviceType::kGPU, dtype>)

REGISTER_ARG_SORT_KERNEL(float);
REGISTER_ARG_SORT_KERNEL(double);
REGISTER_ARG_SORT_KERNEL(int8_t);
REGISTER_ARG_SORT_KERNEL(int32_t);
REGISTER_ARG_SORT_KERNEL(int64_t);

}  // namespace oneflow