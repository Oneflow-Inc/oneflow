#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    fw_func_ = &BoxingKernel<device_type, FloatingPointType>::ConcatBoxForward;
    bw_func_ = &BoxingKernel<device_type, FloatingPointType>::ConcatBoxBackward;
  } else {
    fw_func_ = &BoxingKernel<device_type, FloatingPointType>::AddBoxForward;
    bw_func_ = &BoxingKernel<device_type, FloatingPointType>::AddBoxBackward;
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::InferCopyRules(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto boxing_conf = op()->op_conf().boxing_conf();
  auto in_box_case = boxing_conf.in_box_case();
  // Infer Forward Copy Rules
  if (in_box_case == BoxingOpConf::kConcatBox) {
    // concat-box copy rules: copy directly from input to output
    InferCopyRulesFromBns(BnInOp2Blob, op()->input_bns(), op()->output_bns(),
                          &fw_copy_rules_);
  }
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    InferFwCloneRules(BnInOp2Blob);
  }

  // Infer Backward Copy Rules
  // concat-split box copy rules: copy diffs from odbs to idbs
  if (boxing_conf.out_box_case() == BoxingOpConf::kDataSplitBox) {
    InferCopyRulesFromBns(BnInOp2Blob, op()->input_diff_bns(),
                          op()->output_diff_bns(), &bw_copy_rules_);
    // Reverse back input && output diff blob for each backward rule
    for (CopyRule& rule : bw_copy_rules_) {
      std::swap(rule.src_bn, rule.dst_bn);
      std::swap(rule.src_offset, rule.dst_offset);
    }
  } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    // concat-clone box copy rules: split back to in_diff from middle
    InferCopyRulesFromBns(BnInOp2Blob, {"middle"}, op()->input_diff_bns(),
                          &bw_copy_rules_);
  } else {
    UNEXPECTED_RUN();
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns,
    std::vector<CopyRule>* copy_rules) const {
  // P.S This routine will be called only once, thus some performance
  // loss seems ok.
  std::map<const std::string*, int64_t> src_bn2concat_dim;
  std::map<const std::string*, int64_t> dst_bn2concat_dim;
  int32_t concat_axis = op()->op_conf().boxing_conf().concat_box().axis();
  for (const std::string& bn : src_bns) {
    if (BnInOp2Blob(bn) == nullptr) { break; }
    CHECK(src_bn2concat_dim
              .emplace(&bn, (BnInOp2Blob(bn)->shape().At(concat_axis)))
              .second);
  }
  for (const std::string& bn : dst_bns) {
    if (BnInOp2Blob(bn) == nullptr) { break; }
    CHECK(dst_bn2concat_dim
              .emplace(&bn, (BnInOp2Blob(bn)->shape().At(concat_axis)))
              .second);
  }

  Blob* src_fst_blob = BnInOp2Blob(src_bns.front());
  int64_t concat_dim_sz = src_fst_blob->shape().Count(concat_axis + 1);
  int64_t seg_cnt = (concat_axis == 0) ? 1 : (src_fst_blob->shape().At(0));

  InferCopyRulesFromConcatDim(src_bn2concat_dim, dst_bn2concat_dim, seg_cnt,
                              concat_dim_sz, concat_axis, copy_rules);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::InferCopyRulesFromConcatDim(
    const std::map<const std::string*, int64_t>& src_bn2concat_dim,
    const std::map<const std::string*, int64_t>& dst_bn2concat_dim,
    int64_t seg_cnt, int64_t concat_dim_sz, int32_t concat_axis,
    std::vector<CopyRule>* rules) const {
  int64_t src_offset = 0;
  const int64_t step_sz = sizeof(FloatingPointType);
  for (auto src_iter = src_bn2concat_dim.begin(),
            dst_iter = dst_bn2concat_dim.begin();
       src_iter != src_bn2concat_dim.end()
       && dst_iter != dst_bn2concat_dim.end();) {
    int64_t dst_offset = 0;
    while (dst_offset < dst_iter->second) {
      int64_t p = std::min(src_iter->second - src_offset,
                           dst_iter->second - dst_offset);
      for (size_t i = 0; i < seg_cnt; ++i) {
        CopyRule cr;
        cr.src_bn = *src_iter->first;
        cr.dst_bn = *dst_iter->first;
        cr.src_offset =
            (src_offset + i * src_iter->second) * concat_dim_sz * step_sz;
        cr.dst_offset =
            (dst_offset + i * dst_iter->second) * concat_dim_sz * step_sz;
        cr.copy_sz = p * concat_dim_sz * step_sz;
        rules->push_back(std::move(cr));
      }
      src_offset += p;
      dst_offset += p;
      if (src_offset == src_iter->second) {
        if (++src_iter == src_bn2concat_dim.end()) { break; }
        src_offset = 0;
      }
    }  // while current dst box is not full
    ++dst_iter;
  }
}

// If out box case is a clone box, then previous rules would only encapsulate
// the input source boxes to the first single output box. Hence, we have to
// clone the first output box to the remaining output boxes.
template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::InferFwCloneRules(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const std::vector<std::string>& obns = op()->output_bns();
  int64_t copy_sz = BnInOp2Blob(obns.front())->shape().elem_cnt();
  for (size_t i = 1; i < obns.size(); ++i) {
    if (BnInOp2Blob(obns.at(i)) == nullptr) { break; }
    CopyRule cr;
    cr.src_bn = obns.front();
    cr.dst_bn = obns.at(i);
    cr.src_offset = 0;
    cr.dst_offset = 0;
    cr.copy_sz = copy_sz * sizeof(FloatingPointType);
    fw_copy_rules_.push_back(std::move(cr));
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (fw_copy_rules_.empty()) { InferCopyRules(BnInOp2Blob); }
  (this->*fw_func_)(ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(!bw_copy_rules_.empty());
  (this->*bw_func_)(ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::CopyDataFromRules(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<CopyRule>& copy_rules) const {
  for (const CopyRule& rule : copy_rules) {
    Blob* src_blob = BnInOp2Blob(rule.src_bn);
    Blob* dst_blob = BnInOp2Blob(rule.dst_bn);
    KernelUtil<device_type, FloatingPointType>::Memcpy(
        ctx, dst_blob->mut_dptr<char>() + rule.dst_offset,
        src_blob->dptr<char>() + rule.src_offset, rule.copy_sz);
  }
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::ConcatBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyDataFromRules(ctx, BnInOp2Blob, fw_copy_rules_);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::ConcatBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    // Add all the out-diff blobs into data_tmp blob
    Blob* middle = BnInOp2Blob("middle");
    KernelUtil<device_type, FloatingPointType>::Memset(
        ctx, middle->mut_dptr(), 0,
        middle->shape().elem_cnt() * sizeof(FloatingPointType));
    for (const std::string& bn : op()->output_diff_bns()) {
      Blob* blob = BnInOp2Blob(bn);
      KernelUtil<device_type, FloatingPointType>::BlasAxpy(
          ctx, blob->shape().elem_cnt(), 1.0, blob->dptr<FloatingPointType>(),
          1, middle->mut_dptr<FloatingPointType>(), 1);
    }
  }

  CopyDataFromRules(ctx, BnInOp2Blob, bw_copy_rules_);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::AddBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Copy blob in_0 to out_0
  Blob* out_0 = BnInOp2Blob("out_0");
  Blob* in_0 = BnInOp2Blob("in_0");
  KernelUtil<device_type, FloatingPointType>::Memcpy(
      ctx, out_0->mut_dptr(), in_0->dptr(),
      out_0->shape().elem_cnt() * sizeof(FloatingPointType));
  // Add remaining input blobs to out_0
  for (size_t i = 1; i < op()->input_bns().size(); ++i) {
    Blob* in_i = BnInOp2Blob("in_" + std::to_string(i));
    KernelUtil<device_type, FloatingPointType>::BlasAxpy(
        ctx, out_0->shape().elem_cnt(), 1.0, in_i->dptr<FloatingPointType>(), 1,
        out_0->mut_dptr<FloatingPointType>(), 1);
  }

  // The data in out_0 will be copied to other output blobs.
  CopyDataFromRules(ctx, BnInOp2Blob, fw_copy_rules_);
}

template<DeviceType device_type, typename FloatingPointType>
void BoxingKernel<device_type, FloatingPointType>::AddBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

INSTANTIATE_CPU_KERNEL_CLASS(BoxingKernel);
REGISTER_CPU_KERNEL(OperatorConf::kBoxingConf, BoxingKernel);

}  // namespace oneflow
