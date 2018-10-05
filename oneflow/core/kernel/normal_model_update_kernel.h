#ifndef ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalMdUpdateKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdateKernel);
  virtual ~NormalMdUpdateKernel() = default;

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 protected:
  NormalMdUpdateKernel() = default;
  // void GetTotalInstanceNumFromGpu(std::function<Blob*(const std::string&)> BnInOp2Blob,
  //                                 int32_t* total_instance_num_ptr) const;
  virtual void UpdateModel(DeviceCtx* ctx, const int32_t* total_instance_num_ptr, T learning_rate,
                           T l1, T l2, int64_t next_model_vid,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
};

bool TriggerWarmup(const NormalModelUpdateOpUserConf& conf, double lr, int64_t cur_batch_num);
double GetWarmupLearningRate(const WarmupConf&, double lr, int64_t cur_batch_num);
double GetDecayedLearningRate(const LearningRateDecayConf&, double lr, int64_t cur_batch_num);

#define DECLARE_MDUPDT_KERNEL_CREATOR(x) Kernel* Create##x##MdUpdtKernel(const KernelConf&);

#define DEFINE_MDUPDT_KERNEL_CREATOR(x)                                                  \
  Kernel* Create##x##MdUpdtKernel(const KernelConf& kernel_conf) {                       \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {             \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (x##MdUpdateKernel), \
                                         DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};      \
    return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),    \
                                  kernel_conf.data_type()))();                           \
  }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
