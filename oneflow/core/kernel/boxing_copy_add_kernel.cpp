#include "oneflow/core/kernel/boxing_copy_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BoxingCopyAddKernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {
  const BoxingCopyOpConf& conf = this->op_conf().boxing_copy_conf();
  const TensorPartialView dst_view(conf.out_view());
  for (const TensorPartialViewProto& src_view_proto : conf.in_view()) {
    const TensorPartialView src_view(src_view_proto);
    tensor_partial_copier_vec_.emplace_back(dst_view, src_view, this->kernel_conf().data_type());
  }
}

template<DeviceType device_type, typename T>
void BoxingCopyAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    tensor_partial_copier_vec_.at(i).Exec(ctx.device_ctx, *memory_copier_, out, in_i);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingCopyAddConf, BoxingCopyAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
