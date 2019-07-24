#include "oneflow/core/kernel/sort_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SortKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* key_blob = BnInOp2Blob("key");
  const Blob* value_blob = BnInOp2Blob("value");
  Blob* sorted_key_blob = BnInOp2Blob("sorted_key");
  Blob* sorted_value_blob = BnInOp2Blob("sorted_value");

  const Shape shape = key_blob->shape();
  CHECK_EQ(shape, value_blob->shape());
  CHECK_EQ(shape, sorted_key_blob->shape());
  CHECK_EQ(shape, sorted_value_blob->shape());
  SortUtil<device_type, T>::Forward(
      ctx.device_ctx, key_blob->dptr<T>(), value_blob->dptr<int32_t>(),
      BnInOp2Blob("temp_storage")->mut_dptr<void>(),
      this->op_conf().sort_conf().temp_storage_bytes(), shape.At(0), shape.At(1),
      sorted_key_blob->mut_dptr<T>(), sorted_value_blob->mut_dptr<int32_t>());
}

template<typename T>
struct SortUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const T* key_ptr, const int32_t* value_ptr,
                      void* temp_storage_ptr, size_t temp_storage_bytes, int32_t num_row,
                      int32_t num_col, T* sorted_key_ptr, int32_t* sorted_value_ptr) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSortConf, SortKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
