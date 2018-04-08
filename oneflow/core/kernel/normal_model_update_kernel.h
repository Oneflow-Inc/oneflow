#ifndef ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalMdUpdateKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdateKernel);
  virtual ~NormalMdUpdateKernel() = default;

  void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 protected:
  NormalMdUpdateKernel() = default;
  virtual void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, const Blob* model_diff_blob,
      int64_t next_model_vid, double learning_rate,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

 private:
  Blob* DiffAveragingAndL1Regularization(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

template<DeviceType device_type, typename T>
class NormalMdUpdateKernelUtil final {
 public:
  static void DiffAveragingAndL1Regularization(DeviceCtx* ctx, int64_t n,
                                               float l1, const T* model,
                                               T* model_diff_acc);
};

double GetDecayedLearningRate(const LearningRateDecayConf&, double lr,
                              int64_t now_batch_num);

#define DECLARE_MDUPDT_KERNEL_CREATOR(x) \
  Kernel* Create##x##MdUpdtKernel(const KernelConf&);

#define DEFINE_MDUPDT_KERNEL_CREATOR(x)                                        \
  Kernel* Create##x##MdUpdtKernel(const KernelConf& kernel_conf) {             \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {   \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY,            \
                                         (x##MdUpdateKernel), DEVICE_TYPE_SEQ, \
                                         FLOATING_DATA_TYPE_SEQ)};             \
    return creators.at(                                                        \
        GetHashKey(kernel_conf.device_type(), kernel_conf.data_type()))();     \
  }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
