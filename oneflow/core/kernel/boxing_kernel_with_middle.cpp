#include "oneflow/core/kernel/boxing_kernel_with_middle.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  auto in_box_case = boxing_conf.in_box_case();
  auto out_box_case = boxing_conf.out_box_case();
  if (in_box_case == BoxingOpConf::kConcatBox) {
    if (out_box_case == BoxingOpConf::kDataSplitBox) {
      fw_func_ =
          &BoxingKernel<device_type, FloatingPointType>::ConcatSplitBoxForward;
      bw_func_ =
          &BoxingKernel<device_type, FloatingPointType>::ConcatSplitBoxBackward;
    } else {
      fw_func_ =
          &BoxingKernel<device_type, FloatingPointType>::ConcatCloneBoxForward;
      bw_func_ =
          &BoxingKernel<device_type, FloatingPointType>::ConcatCloneBoxBackward;
    }
  } else {
    fw_func_ =
        &BoxingKernel<device_type, FloatingPointType>::AddCloneBoxForward;
    bw_func_ =
        &BoxingKernel<device_type, FloatingPointType>::AddCloneBoxBackward;
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::AddBoxes2Middle(
    const KernelCtx& ctx, const std::vector<std::string>& bns,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  // Copy first blob in bns to middle
  Blob* b_0 = BnInOp2BlobPtr(bns.at(0));
  Blob* middle = BnInOp2BlobPtr("middle");
  KernelUtil<device_type, FloatingPointType>::Memcpy(
      ctx, middle->mut_dptr(), b_0->dptr(),
      middle->shape().elem_cnt() * sizeof(FloatingPointType));

  // Add remaining blobs in bns to middle
  for (size_t i = 0; i < bns.size(); ++i) {
    Blob* b_i = BnInOp2BlobPtr(bns.at(i));
    KernelUtil<device_type, FloatingPointType>::BlasAxpy(
        ctx, middle->shape().elem_cnt(), 1.0,
        static_cast<const FloatingPointType*>(b_i->dptr()), 1,
        static_cast<FloatingPointType*>(middle->mut_dptr()), 1);
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::SplitMiddle2Boxes(
    const KernelCtx& ctx, const std::vector<std::string>& bns,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    bool reverse) const {
  Blob* middle = BnInOp2BlobPtr("middle");
  const int32_t concat_axis = op()->op_conf().boxing_conf().concat_box().axis();
  const int64_t middle_sz =
      middle->shape().elem_cnt() * sizeof(FloatingPointType);
  const int64_t step_sz = middle->shape().Count(1);
  const int64_t step_num = (concat_axis == 0) ? 1 : middle->shape().At(0);
  for (size_t i = 0; i < step_num; ++i) {
    size_t j = 0;
    for (int64_t offset = step_sz * i; offset < middle_sz;) {
      Blob* b_j = BnInOp2BlobPtr(bns.at(j));
      int64_t b_sz = b_j->shape().elem_cnt() * sizeof(FloatingPointType);
      if (!reverse) {
        KernelUtil<device_type, FloatingPointType>::Memcpy(
            ctx, b_j->mut_dptr(),
            static_cast<const char*>(middle->dptr()) + offset, b_sz);
      } else {
        KernelUtil<device_type, FloatingPointType>::Memcpy(
            ctx, static_cast<char*>(middle->mut_dptr()) + offset, b_j->dptr(),
            b_sz);
      }
      offset += b_sz;
    }
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::CopyMiddle2Boxes(
    const KernelCtx& ctx, const std::vector<std::string>& bns,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* middle = BnInOp2BlobPtr("middle");
  int64_t middle_sz = middle->shape().elem_cnt() * sizeof(FloatingPointType);
  for (const std::string& bn : bns) {
    Blob* out_i = BnInOp2BlobPtr(bn);
    KernelUtil<device_type, FloatingPointType>::Memcpy(
        ctx, out_i->mut_dptr(), middle->dptr(), middle_sz);
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  (this->*fw_func_)(ctx, BnInOp2BlobPtr);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  (this->*bw_func_)(ctx, BnInOp2BlobPtr);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::ConcatSplitBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  SplitMiddle2Boxes(ctx, op()->input_bns(), BnInOp2BlobPtr, true);
  SplitMiddle2Boxes(ctx, op()->output_bns(), BnInOp2BlobPtr, false);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::ConcatSplitBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  AddBoxes2Middle(ctx, op()->output_diff_bns(), BnInOp2BlobPtr);
  SplitMiddle2Boxes(ctx, op()->input_diff_bns(), BnInOp2BlobPtr, false);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::ConcatCloneBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  SplitMiddle2Boxes(ctx, op()->input_bns(), BnInOp2BlobPtr, true);
  // Clone middle blob to output blobs
  CopyMiddle2Boxes(ctx, op()->output_bns(), BnInOp2BlobPtr);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::ConcatCloneBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  AddBoxes2Middle(ctx, op()->output_diff_bns(), BnInOp2BlobPtr);
  SplitMiddle2Boxes(ctx, op()->input_diff_bns(), BnInOp2BlobPtr, false);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::AddCloneBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  AddBoxes2Middle(ctx, op()->input_bns(), BnInOp2BlobPtr);
  CopyMiddle2Boxes(ctx, op()->output_bns(), BnInOp2BlobPtr);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::AddCloneBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  UNEXPECTED_RUN();
}

INSTANTIATE_CPU_KERNEL_CLASS(BoxingKernel);
REGISTER_CPU_KERNEL(OperatorConf::kBoxingConf, BoxingKernel);

}  // namespace oneflow
