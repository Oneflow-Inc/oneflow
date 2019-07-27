#include "oneflow/core/kernel/sort_kernel.h"

namespace oneflow {

template<typename T>
void CpuSort(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num, int32_t instance_size,
             std::string dir, T* out_ptr) {
  FOR_RANGE(int32_t, i, 0, instance_num) {
    // TODO
  }
}

template<DeviceType device_type, typename T>
void SortKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  int32_t instance_size = in_blob->shape().dim_vec().back();
  int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
  const T* in_ptr = in_blob->dptr<T>();
  T* out_ptr = out_blob->mut_dptr<T>();

  if (this->op_conf().device_type() == DeviceType::kCPU) {
    CpuSort(ctx.device_ctx, in_ptr, instance_num, instance_size, this->op_conf().sort_conf().dir(),
            out_ptr);
  } else if (this->op_conf().device_type() == DeviceType::kGPU) {
    GpuSort(ctx.device_ctx, in_ptr, instance_num, instance_size, this->op_conf().sort_conf().dir(),
            BnInOp2Blob("temp_storage")->mut_dptr<void>(),
            this->kernel_conf().sort_conf().temp_storage_bytes(), out_ptr);
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSortConf, SortKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
