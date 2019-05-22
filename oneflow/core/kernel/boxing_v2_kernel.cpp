#include "oneflow/core/kernel/boxing_v2_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void BoxingV2Kernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {
  memory_copier_.reset(NewDefaultMemoryCopier(device_type));
  const BoxingV2Conf& conf = GetCustomizedBoxingConf();
  const TensorPartialView dst_view(conf.out_view());
  for (const TensorPartialViewProto& src_view_proto : conf.in_view()) {
    const TensorPartialView src_view(src_view_proto);
    tensor_partial_copier_vec_.emplace_back(
        new TensorPartialCopier(dst_view, src_view, this->kernel_conf().data_type()));
  }
}

template<DeviceType device_type, typename T>
MemoryCopier* BoxingV2Kernel<device_type, T>::memory_copier() const {
  return memory_copier_.get();
}

template<DeviceType device_type, typename T>
const std::vector<std::shared_ptr<TensorPartialCopier>>&
BoxingV2Kernel<device_type, T>::tensor_partial_copier_vec() const {
  return tensor_partial_copier_vec_;
}

template<DeviceType device_type, typename T>
const BoxingV2Conf& BoxingV2CopyKernel<device_type, T>::GetCustomizedBoxingConf() const {
  return this->op_conf().boxing_v2_copy_conf().boxing_conf();
}

template<DeviceType device_type, typename T>
void BoxingV2CopyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    this->tensor_partial_copier_vec().at(i)->Exec(ctx.device_ctx, *this->memory_copier(), out,
                                                  in_i);
  }
}

template<DeviceType device_type, typename T>
const BoxingV2Conf& BoxingV2AddKernel<device_type, T>::GetCustomizedBoxingConf() const {
  return this->op_conf().boxing_v2_add_conf().boxing_conf();
}

template<DeviceType device_type, typename T>
void BoxingV2AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  Blob* buf = BnInOp2Blob("buf");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    if (i == 0) {
      this->tensor_partial_copier_vec().at(i)->Exec(ctx.device_ctx, *this->memory_copier(), out,
                                                    in_i);
    } else {
      this->tensor_partial_copier_vec().at(i)->Exec(ctx.device_ctx, *this->memory_copier(), buf,
                                                    in_i);
      Addition<device_type, T>(ctx.device_ctx, out, out, buf);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingV2CopyConf, BoxingV2CopyKernel, POD_DATA_TYPE_SEQ)
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingV2AddConf, BoxingV2AddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
