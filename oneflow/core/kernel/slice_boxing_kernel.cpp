#include "oneflow/core/kernel/slice_boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void SliceBoxingKernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {
  memory_copier_.reset(NewDefaultMemoryCopier(device_type));
  const SliceBoxingConf& conf = GetCustomizedBoxingConf();
  const TensorSliceView out_slice(conf.out_slice());
  for (const TensorSliceViewProto& in_slice_proto : conf.in_slice()) {
    const TensorSliceView in_slice(in_slice_proto);
    tensor_slice_copier_vec_.emplace_back(
        new TensorSliceCopier(out_slice, in_slice, this->kernel_conf().data_type()));
  }
}

template<DeviceType device_type, typename T>
MemoryCopier* SliceBoxingKernel<device_type, T>::memory_copier() const {
  return memory_copier_.get();
}

template<DeviceType device_type, typename T>
const std::vector<std::shared_ptr<TensorSliceCopier>>&
SliceBoxingKernel<device_type, T>::tensor_slice_copier_vec() const {
  return tensor_slice_copier_vec_;
}

template<DeviceType device_type, typename T>
const SliceBoxingConf& SliceBoxingCopyKernel<device_type, T>::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_copy_conf().slice_boxing_conf();
}

template<DeviceType device_type, typename T>
void SliceBoxingCopyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), out, in_i);
  }
}

template<DeviceType device_type, typename T>
const SliceBoxingConf& SliceBoxingAddKernel<device_type, T>::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_add_conf().slice_boxing_conf();
}

template<DeviceType device_type, typename T>
void SliceBoxingAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  Blob* buf = BnInOp2Blob("buf");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    if (i == 0) {
      this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), out,
                                                  in_i);
    } else {
      this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), buf,
                                                  in_i);
      Addition<device_type, T>(ctx.device_ctx, out, out, buf);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyKernel,
                           POD_DATA_TYPE_SEQ)
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
