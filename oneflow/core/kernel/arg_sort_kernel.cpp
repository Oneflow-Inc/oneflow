#include "oneflow/core/kernel/arg_sort_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ArgSortKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  CHECK(in_blob->has_dim0_valid_num_field());
  const int32_t dim0_valid_num = in_blob->dim0_valid_num(0);
  if (device_type == DeviceType::kGPU) {
    BnInOp2Blob("indices")->set_dim0_valid_num(0, dim0_valid_num);
    BnInOp2Blob("sorted_in")->set_dim0_valid_num(0, dim0_valid_num);
  }
  BnInOp2Blob("out")->set_dim0_valid_num(0, dim0_valid_num);
}

template<DeviceType device_type, typename T>
void ArgSortKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  CHECK(in_blob->has_instance_shape_field());
  const Shape instance_shape = in_blob->instance_shape();
  if (device_type == DeviceType::kGPU) {
    BnInOp2Blob("indices")->set_instance_shape(instance_shape);
    BnInOp2Blob("sorted_in")->set_instance_shape(instance_shape);
  }
  BnInOp2Blob("out")->set_instance_shape(instance_shape);
}

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
      if (dir == "ASCENDING") {
        return l < r;
      } else if (dir == "DESCENDING") {
        return l > r;
      } else {
        UNIMPLEMENTED();
      }
    };
    std::stable_sort(out_ptr_i, out_ptr_i + instance_size, comp);
  }
}

template<DeviceType device_type, typename T>
void ArgSortKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
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

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kArgSortConf, ArgSortKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
