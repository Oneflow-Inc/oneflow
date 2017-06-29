#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    fw_func_ = &BoxingKernel<device_type,
             floating_point_type>::ConcatBoxForward;
    bw_func_ = &BoxingKernel<device_type,
             floating_point_type>::ConcatBoxBackward;
  } else {
    fw_func_ = &BoxingKernel<device_type,
             floating_point_type>::AddBoxForward;
    bw_func_ = &BoxingKernel<device_type,
             floating_point_type>::AddBoxBackward;
  }
}

// Infer all the copy rules during execution time.
template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::InferCopyRules(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  auto boxing_conf = op()->op_conf().boxing_conf();
  auto in_box_case = boxing_conf.in_box_case();
  // Infer Forward Copy Rules
  if (in_box_case == BoxingOpConf::kAddBox) {
    // add-box copy rules: copy from middle to output
    InferCopyRulesFromBns(BnInOp2BlobPtr, {"middle"}, op()->output_bns(),
                          &fw_copy_rules);
  } else if (in_box_case == BoxingOpConf::kConcatBox) {
    // concat-box copy rules: copy directly from input to output
    InferCopyRulesFromBns(BnInOp2BlobPtr, op()->input_bns(),
                          op()->output_bns(), &fw_copy_rules);
  } else {
    // do nothing
  }
  // Mark: if output is clone-box, add forward copy rules from first 
  // output blob to the remaining blobs
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    ConstructFwCloneRules(BnInOp2BlobPtr);
  }

  // Infer Backward Copy Rules
  // concat-split box copy rules: directly decompose && assemble
  if (boxing_conf.out_box_case() == BoxingOpConf::kDataSplitBox) {
    InferCopyRulesFromBns(BnInOp2BlobPtr, op()->input_diff_bns(),
                          op()->output_diff_bns(), &bw_copy_rules);
    // Reverse back input && output diff blob for each backward rule
    for (auto& rule : bw_copy_rules) {
      std::swap(rule.src_bn, rule.dst_bn);
      std::swap(rule.src_offset, rule.dst_offset);
    }
  } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    // concat-clone box copy rules: split back to in_diff from middle
    InferCopyRulesFromBns(BnInOp2BlobPtr, {"middle"}, op()->input_diff_bns(),
                          &bw_copy_rules);
  } else {
    // do nothing
  }
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns,
    std::vector<copy_rule>* copy_rules) const {

  // P.S This routine will be called only once, thus some performance
  // loss seems ok.
  std::map<const std::string*, int64_t> src_bn2slice; 
  std::map<const std::string*, int64_t> dst_bn2slice; 
  int32_t concat_axis = op()->op_conf().boxing_conf().concat_box().axis();
  for (const std::string& bn : src_bns) {
    CHECK(src_bn2slice.emplace(std::make_pair(&bn,
          (BnInOp2BlobPtr(bn)->shape().At(concat_axis)))).second);
  }
  for (const std::string& bn : dst_bns) {
    CHECK(dst_bn2slice.emplace(std::make_pair(&bn,
          (BnInOp2BlobPtr(bn)->shape().At(concat_axis)))).second);
  }
  
  Blob* src_fst_blob = BnInOp2BlobPtr(src_bns.front());
  int64_t slice_sz = src_fst_blob->shape().Count(concat_axis+1);
  int64_t seg_cnt = (concat_axis == 0) ? 1 : (src_fst_blob->shape().At(0));

  ConstructCopyRulesFromSlice(src_bn2slice, dst_bn2slice, seg_cnt, 
                              slice_sz, concat_axis, copy_rules);
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type,
                  floating_point_type>::ConstructCopyRulesFromSlice(
  const std::map<const std::string*, int64_t>& src_bn2slice, 
  const std::map<const std::string*, int64_t>& dst_bn2slice,
  int64_t seg_cnt, int64_t slice_sz, int32_t concat_axis, 
  std::vector<struct copy_rule>* rules) const {
  auto src_iter = src_bn2slice.begin();
  auto dst_iter = dst_bn2slice.begin();
  int64_t src_offset = 0, src_cap = src_iter->second;
  int64_t dst_offset = 0, dst_cap = dst_iter->second;

  const int64_t step_sz = sizeof(floating_point_type);
  while (src_iter != src_bn2slice.end() 
      && dst_iter != dst_bn2slice.end()) {
    dst_offset = 0, dst_cap = dst_iter->second;
    while (dst_offset < dst_cap) {
      int64_t p = std::min(src_cap-src_offset, dst_cap-dst_offset);
      for (size_t i=0; i < seg_cnt; ++i) {
        struct copy_rule cr;
        cr.src_bn = *src_iter->first;
        cr.dst_bn = *dst_iter->first;
        cr.src_offset = (src_offset + i * src_cap) * slice_sz * step_sz;
        cr.dst_offset = (dst_offset + i * dst_cap) * slice_sz * step_sz;
        cr.copy_sz = p * slice_sz * step_sz;
        rules->push_back(std::move(cr));
      }
      src_offset += p, dst_offset += p;
      if (src_offset == src_cap) {
        if (++src_iter == src_bn2slice.end()) { 
          break;
        }
        src_offset = 0;
        src_cap = src_iter->second;
      }
    }  // while current dst_cap is not full
    ++dst_iter;
  }
}

// If out box case is a clone box, then previous rules would only encapsulate 
// the input source boxes to the first single output box. Hence, we have to 
// clone the first output box to the remaining output boxes.
template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::ConstructFwCloneRules(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::vector<std::string>& obns = op()->output_bns();
  int64_t copy_sz = BnInOp2BlobPtr(obns.front())->shape().elem_cnt();
  for (size_t i=1; i < obns.size(); ++i) {
    struct copy_rule cr;
    cr.src_bn = obns.front();
    cr.dst_bn = obns.at(i);
    cr.src_offset = 0;
    cr.dst_offset = 0;
    cr.copy_sz = copy_sz * sizeof(floating_point_type);
    fw_copy_rules.push_back(std::move(cr));
  }
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  if (fw_copy_rules.empty()) {
    InferCopyRules(BnInOp2BlobPtr);
  }
  (this->*fw_func_)(ctx, BnInOp2BlobPtr); 
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  CHECK(!bw_copy_rules.empty());
  if (bw_copy_rules.empty()) {
    InferCopyRules(BnInOp2BlobPtr);
  }
  (this->*bw_func_)(ctx, BnInOp2BlobPtr);
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::CopyDataFromRules(
    const KernelCtx& ctx, 
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr, 
    const std::vector<copy_rule>& copy_rules) const {
  for (auto rule : copy_rules) {
    Blob* src_blob = BnInOp2BlobPtr(rule.src_bn);
    Blob* dst_blob = BnInOp2BlobPtr(rule.dst_bn);
    KernelUtil<device_type, floating_point_type>::Memcpy(ctx, 
        static_cast<char*>(dst_blob->mut_dptr())
        + rule.dst_offset,
        static_cast<const char*>(src_blob->dptr())
        + rule.src_offset, rule.copy_sz);
  }
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::ConcatBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
    CopyDataFromRules(ctx, BnInOp2BlobPtr, fw_copy_rules);
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::ConcatBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
    auto boxing_conf = op()->op_conf().boxing_conf();
    if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      // Add all the out-diff blobs into data_tmp blob
      Blob* middle = BnInOp2BlobPtr("middle");
      KernelUtil<device_type, floating_point_type>::Memset(ctx, 
          middle->mut_dptr(), 0, middle->shape().elem_cnt() * 
          sizeof(floating_point_type));
      for (auto bn : op()->output_diff_bns()) {
        Blob* blob = BnInOp2BlobPtr(bn);
        KernelUtil<device_type, floating_point_type>::BlasAxpy(ctx, 
            blob->shape().elem_cnt(), 1.0, 
            static_cast<const floating_point_type*>(blob->dptr()), 1,
            static_cast<floating_point_type*>(middle->mut_dptr()), 1);
      }
    }

    CopyDataFromRules(ctx, BnInOp2BlobPtr, bw_copy_rules);
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::AddBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
    // Add all the input blobs into data_tmp blob
    Blob* middle = BnInOp2BlobPtr("middle");
    KernelUtil<device_type, floating_point_type>::Memset(ctx, 
        middle->mut_dptr(), 0, middle->shape().elem_cnt() * 
        sizeof(floating_point_type));
    for (auto bn : op()->input_bns()) {
      Blob* blob = BnInOp2BlobPtr(bn);
      KernelUtil<device_type, floating_point_type>::BlasAxpy(ctx, 
          BnInOp2BlobPtr(bn)->shape().elem_cnt(), 1.0, 
          static_cast<const floating_point_type*>(blob->dptr()), 1,
          static_cast<floating_point_type*>(middle->mut_dptr()), 1);
    }

    // look at the copy table, transfer the data from data_tmp
    CopyDataFromRules(ctx, BnInOp2BlobPtr, fw_copy_rules); 
}

template<DeviceType device_type, typename floating_point_type>
void BoxingKernel<device_type, floating_point_type>::AddBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  // Discussed with Will, currently no backward actions of addbox
  // do nothing
  UNEXPECTED_RUN();
}

INSTANTIATE_CPU_KERNEL_CLASS(BoxingKernel);
REGISTER_CPU_KERNEL(OperatorConf::kBoxingConf, BoxingKernel);

}  // namespace oneflow
