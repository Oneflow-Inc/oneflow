#ifndef ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumMdUpdateKernel);
  MomentumMdUpdateKernel() = default;
  ~MomentumMdUpdateKernel() = default;

 protected:
  void MemSetMovingModelBlobs(DeviceCtx* ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitMovingModelBlobsWithDir(
      DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  void UpdateModel(DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2,
                   const Blob* pre_model_blob, const Blob* model_diff_blob, int64_t next_model_vid,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, int64_t batch_size, T beta, T learning_rate, T l1,
                          T l2, const T* model_diff, const T* pre_model, T* momentum, T* model);
};

DECLARE_MDUPDT_KERNEL_CREATOR(Momentum);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
