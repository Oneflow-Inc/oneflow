#include "oneflow/core/kernel/naive_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

template<DeviceType device_type, typename T>
void NaiveMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T learning_rate, T l1, T l2, int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  // model = model - alpha * model_diff
  NaiveMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, l1, l2, model_diff_blob->dptr<T>(),
      model_blob->mut_dptr<T>());
}

template<typename T>
class NaiveMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, const int64_t n, T learning_rate, T l1, T l2,
                          const T* model_diff, T* model) {
    for (int64_t i = 0; i != n; ++i) {
      T reg_diff = RegularizeDiff(model_diff[i], l1, l2, model[i]);
      model[i] = model[i] - learning_rate * reg_diff;
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Naive);

}  // namespace oneflow
