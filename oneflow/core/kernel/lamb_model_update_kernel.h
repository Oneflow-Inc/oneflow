#ifndef ONEFLOW_CORE_KERNEL_LAMB_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAMB_MODEL_UDPATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LAMBMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LAMBMdUpdateKernel);
  LAMBMdUpdateKernel() = default;
  ~LAMBMdUpdateKernel() = default;

 private:
  void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void UpdateModel(DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
                   int64_t next_model_vid,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class LAMBMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const T* batch_instance_num_ptr, T learning_rate,
                          T l1, T l2, float beta1, float beta2, float epsilon,
                          int64_t next_model_vid, const float* beta1_t, const float* beta2_t,
                          T* model_diff, T* model, T* m, T* v, T* fw_buf);
};

DECLARE_MDUPDT_KERNEL_CREATOR(LAMB);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAMB_MODEL_UPDATE_KERNEL_H_
