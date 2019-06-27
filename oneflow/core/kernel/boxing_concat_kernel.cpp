#include "oneflow/core/kernel/boxing_concat_kernel.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/register/tensor_slice_copier.h"

namespace oneflow {

template<DeviceType device_type>
void BoxingConcatKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingConcatOpConf& conf = this->op_conf().boxing_concat_conf();
  if (conf.axis() == 0) {
    int64_t dim0_valid_num = 0;
    FOR_RANGE(int64_t, i, 0, conf.in_num()) {
      dim0_valid_num += BnInOp2Blob(GenRepeatedBn("in", i))->shape().At(0);
    }
    BnInOp2Blob("out")->set_dim0_valid_num(0, dim0_valid_num);
  } else {
    const int64_t dim0_valid_num = BnInOp2Blob(GenRepeatedBn("in", 0))->shape().At(0);
    FOR_RANGE(int64_t, i, 1, conf.in_num()) {
      CHECK_EQ(BnInOp2Blob(GenRepeatedBn("in", i))->shape().At(i), dim0_valid_num);
    }
    BnInOp2Blob("out")->set_dim0_valid_num(0, dim0_valid_num);
  }
}

template<DeviceType device_type>
void BoxingConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t offset = 0;
  Blob* out = BnInOp2Blob("out");
  const TensorSliceView out_slice(out->shape());
  const BoxingConcatOpConf& conf = this->op_conf().boxing_concat_conf();
  std::vector<Range> in_ranges = out_slice.range_vec();
  const int64_t axis = conf.axis();
  std::unique_ptr<MemoryCopier> memory_copier(NewDefaultMemoryCopier(device_type));
  FOR_RANGE(int64_t, i, 0, conf.in_num()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    in_ranges[axis].mut_begin() = offset;
    in_ranges[axis].mut_end() = offset + in_i->shape().At(axis);
    offset += in_i->shape().At(axis);
    const TensorSliceView in_slice(in_ranges);
    const TensorSliceCopier slice_copier(out_slice, in_slice, out->data_type());
    slice_copier.Copy(ctx.device_ctx, *memory_copier, out, in_i);
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kBoxingConcatConf, BoxingConcatKernel)

}  // namespace oneflow
