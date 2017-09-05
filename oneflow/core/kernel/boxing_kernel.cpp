#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<typename T>
void BoxingKernel<T>::InitFromOpProto(const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    fw_func_ = &BoxingKernel<T>::ConcatBoxForward;
    bw_func_ = &BoxingKernel<T>::ConcatBoxBackward;
  } else {
    fw_func_ = &BoxingKernel<T>::AddBoxForward;
    bw_func_ = &BoxingKernel<T>::AddBoxBackward;
  }
}

template<typename T>
void BoxingKernel<T>::InferFwCopyRules(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto boxing_conf = op()->op_conf().boxing_conf();
  auto in_box_case = boxing_conf.in_box_case();
  if (in_box_case == BoxingOpConf::kConcatBox) {
    // concat-box copy rules: copy directly from input to output
    int32_t concat_axis = boxing_conf.concat_box().axis();
    if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      InferCopyRulesFromBns(BnInOp2Blob, op()->input_bns(), {"out_0"},
                            concat_axis, 0, &fw_copy_rules_);
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kDataSplitBox) {
      InferCopyRulesFromBns(BnInOp2Blob, op()->input_bns(), op()->output_bns(),
                            concat_axis, 0, &fw_copy_rules_);
    } else {
      UNEXPECTED_RUN();
    }
  }
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    InferFwCloneRules(BnInOp2Blob);
  }
}

template<typename T>
void BoxingKernel<T>::InferBwCopyRules(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // concat-split box copy rules: copy diffs from odbs to idbs
  auto boxing_conf = op()->op_conf().boxing_conf();
  int32_t concat_axis = boxing_conf.concat_box().axis();
  if (boxing_conf.out_box_case() == BoxingOpConf::kDataSplitBox) {
    InferCopyRulesFromBns(BnInOp2Blob, op()->input_diff_bns(),
                          op()->output_diff_bns(), concat_axis, 0,
                          &bw_copy_rules_);
    // Reverse back input && output diff blob for each backward rule
    for (CopyRule& rule : bw_copy_rules_) {
      std::swap(rule.src_bn, rule.dst_bn);
      std::swap(rule.src_offset, rule.dst_offset);
    }
  } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    // concat-clone box copy rules: split back to in_diff from middle
    InferCopyRulesFromBns(BnInOp2Blob, {"middle"}, op()->input_diff_bns(), 0,
                          concat_axis, &bw_copy_rules_);
  } else {
    UNEXPECTED_RUN();
  }
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns, const int32_t src_concat_axis,
    const int32_t dst_split_axis, std::vector<CopyRule>* copy_rules) const {
  std::vector<Blob*> src_blobs;
  std::vector<Blob*> dst_blobs;
  for (const std::string& bn : src_bns) {
    Blob* b = BnInOp2Blob(bn);
    if (b == nullptr) { break; }
    src_blobs.emplace_back(b);
  }
  for (const std::string& bn : dst_bns) {
    Blob* b = BnInOp2Blob(bn);
    if (b == nullptr) { break; }
    dst_blobs.emplace_back(b);
  }
  InferDataIdCopyRules(src_bns, dst_bns, src_blobs, dst_blobs, src_concat_axis,
                       dst_split_axis, copy_rules);

  if (src_concat_axis == dst_split_axis) {
    InferCopyRulesFromEqualAxis(BnInOp2Blob, src_bns, dst_bns, copy_rules);
  } else {
    InferCopyRulesFromUnequalAxis(src_bns, dst_bns, src_blobs, dst_blobs,
                                  src_concat_axis, dst_split_axis, copy_rules);
  }
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromUnequalAxis(
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns, const std::vector<Blob*> src_blobs,
    const std::vector<Blob*> dst_blobs, const int32_t src_concat_axis,
    const int32_t dst_split_axis, std::vector<CopyRule>* copy_rules) const {
  const int64_t im_sz = (src_blobs.at(0)->shape().NumAxes() > 2)
                            ? src_blobs.at(0)->shape().Count(2) * sizeof(T)
                            : sizeof(T);
  if (src_concat_axis == 0 && dst_split_axis == 1) {
    int64_t row_pre_offset_sum = 0;
    for (size_t src_idx = 0; src_idx < src_blobs.size(); ++src_idx) {
      Blob* src_b = src_blobs.at(src_idx);
      for (size_t offset = 0; offset < src_b->shape().At(0); ++offset) {
        int64_t col_pre_offset_sum = 0;
        for (size_t dst_idx = 0; dst_idx < dst_blobs.size(); ++dst_idx) {
          Blob* dst_b = dst_blobs.at(dst_idx);
          CopyRule cr;
          cr.src_bn = src_bns.at(src_idx);
          cr.dst_bn = dst_bns.at(dst_idx);
          cr.src_offset =
              (offset * src_b->shape().At(1) + col_pre_offset_sum) * im_sz;
          cr.dst_offset =
              ((row_pre_offset_sum + offset) * dst_b->shape().At(1)) * im_sz;
          cr.copy_sz = dst_b->shape().At(1) * im_sz;
          copy_rules->push_back(std::move(cr));

          col_pre_offset_sum += dst_b->shape().At(1);
        }
      }
      row_pre_offset_sum += src_b->shape().At(0);
    }
  } else if (src_concat_axis == 1 && dst_split_axis == 0) {
    int64_t col_pre_offset_sum = 0;
    for (size_t src_idx = 0; src_idx < src_blobs.size(); ++src_idx) {
      Blob* src_b = src_blobs.at(src_idx);
      int64_t row_pre_offset_sum = 0;
      for (size_t dst_idx = 0; dst_idx < dst_blobs.size(); ++dst_idx) {
        Blob* dst_b = dst_blobs.at(dst_idx);
        for (size_t offset = 0; offset < dst_b->shape().At(0); ++offset) {
          CopyRule cr;
          cr.src_bn = src_bns.at(src_idx);
          cr.dst_bn = dst_bns.at(dst_idx);
          cr.src_offset =
              ((row_pre_offset_sum + offset) * src_b->shape().At(1)) * im_sz;
          cr.dst_offset =
              (offset * dst_b->shape().At(1) + col_pre_offset_sum) * im_sz;
          cr.copy_sz = src_b->shape().At(1) * im_sz;
          copy_rules->push_back(std::move(cr));
        }
        row_pre_offset_sum += dst_b->shape().At(0);
      }
      col_pre_offset_sum += src_b->shape().At(1);
    }
  } else {
    UNEXPECTED_RUN();
  }
}

template<typename T>
void BoxingKernel<T>::InferDataIdCopyRules(
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns, const std::vector<Blob*> src_blobs,
    const std::vector<Blob*> dst_blobs, const int32_t src_concat_axis,
    const int32_t dst_split_axis, std::vector<CopyRule>* rules) const {
  if (src_concat_axis == 0) {
    int64_t src_idx = 0;
    int64_t dst_idx = 0;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    while (src_idx < src_blobs.size() && dst_idx < dst_blobs.size()) {
      int64_t src_cap = src_blobs.at(src_idx)->shape().At(0);
      int64_t dst_cap = dst_blobs.at(dst_idx)->shape().At(0);
      int64_t q = std::min(src_cap - src_offset, dst_cap - dst_offset);

      CopyRule rule;
      rule.src_bn = src_bns.at(src_idx);
      rule.dst_bn = dst_bns.at(dst_idx);
      rule.src_offset = src_offset * sizeof(T);
      rule.dst_offset = dst_offset * sizeof(T);
      rule.copy_sz = q * sizeof(T);
      rules->push_back(std::move(rule));

      src_offset += q;
      if (src_offset == src_cap) {
        src_offset = 0;
        ++src_idx;
      }

      dst_offset += q;
      if (dst_offset == dst_cap) {
        dst_offset = 0;
        ++dst_idx;
      }
    }
  } else if (src_concat_axis == 1) {
    CopyRule rule;
    rule.src_bn = src_bns.at(0);
    rule.dst_bn = dst_bns.at(0);
    rule.src_offset = 0;
    rule.dst_offset = 0;
    rule.copy_sz = src_blobs.at(0)->ByteSizeOfDataIdField();
    rules->push_back(std::move(rule));
  } else {
    UNEXPECTED_RUN();
  }

  if (dst_split_axis == 1) {
    // add copy rules from first dst blob to all dst blobs
    for (size_t i = 1; i < dst_bns.size(); ++i) {
      CopyRule rule;
      rule.src_bn = dst_bns.at(0);
      rule.dst_bn = dst_bns.at(i);
      rule.src_offset = 0;
      rule.dst_offset = 0;
      rule.copy_sz = dst_blobs.at(0)->ByteSizeOfDataIdField();
      rules->push_back(std::move(rule));
    }
  }
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromEqualAxis(
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
  const int64_t concat_dim_sz =
      (src_fst_blob->shape().NumAxes() > concat_axis + 1)
          ? src_fst_blob->shape().Count(concat_axis + 1)
          : 1;
  int64_t seg_cnt = (concat_axis == 0) ? 1 : (src_fst_blob->shape().At(0));

  InferCopyRulesFromConcatDim(src_bn2concat_dim, dst_bn2concat_dim, seg_cnt,
                              concat_dim_sz, concat_axis, copy_rules);
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromConcatDim(
    const std::map<const std::string*, int64_t>& src_bn2concat_dim,
    const std::map<const std::string*, int64_t>& dst_bn2concat_dim,
    int64_t seg_cnt, int64_t concat_dim_sz, int32_t concat_axis,
    std::vector<CopyRule>* rules) const {
  int64_t src_offset = 0;
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
            (src_offset + i * src_iter->second) * concat_dim_sz * sizeof(T);
        cr.dst_offset =
            (dst_offset + i * dst_iter->second) * concat_dim_sz * sizeof(T);
        cr.copy_sz = p * concat_dim_sz * sizeof(T);
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
template<typename T>
void BoxingKernel<T>::InferFwCloneRules(
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
    cr.copy_sz = copy_sz * sizeof(T);
    fw_copy_rules_.push_back(std::move(cr));
  }
}

template<typename T>
void BoxingKernel<T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (fw_copy_rules_.empty()) { InferFwCopyRules(BnInOp2Blob); }
  (this->*fw_func_)(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (bw_copy_rules_.empty()) { InferBwCopyRules(BnInOp2Blob); }
  (this->*bw_func_)(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::CopyDataFromRules(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<CopyRule>& copy_rules) const {
  for (const CopyRule& rule : copy_rules) {
    Blob* src_blob = BnInOp2Blob(rule.src_bn);
    Blob* dst_blob = BnInOp2Blob(rule.dst_bn);
    Memcpy<DeviceType::kCPU>(
        ctx.device_ctx, dst_blob->mut_dptr<char>() + rule.dst_offset,
        src_blob->dptr<char>() + rule.src_offset, rule.copy_sz,
        cudaMemcpyKind::cudaMemcpyHostToHost);
  }
}

template<typename T>
void BoxingKernel<T>::ConcatBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyDataFromRules(ctx, BnInOp2Blob, fw_copy_rules_);
}

template<typename T>
void BoxingKernel<T>::AddBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

#define FLOATING_BOXING_KERNEL_CONCAT_BOX_BACKWARD(type_cpp, type_proto) \
  template<>                                                             \
  void BoxingKernel<type_cpp>::ConcatBoxBackward(                        \
      const KernelCtx& ctx,                                              \
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {      \
    auto boxing_conf = op()->op_conf().boxing_conf();                    \
    if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {         \
      Blob* middle = BnInOp2Blob("middle");                              \
      Memset<DeviceType::kCPU>(ctx.device_ctx, middle->mut_dptr(), 0,    \
                               middle->ByteSizeOfDataField());           \
      for (const std::string& bn : op()->output_diff_bns()) {            \
        Blob* blob = BnInOp2Blob(bn);                                    \
        KernelUtil<DeviceType::kCPU, type_cpp>::BlasAxpy(                \
            ctx.device_ctx, blob->shape().elem_cnt(),                    \
            static_cast<type_cpp>(1.0), blob->dptr<type_cpp>(), 1,       \
            middle->mut_dptr<type_cpp>(), 1);                            \
      }                                                                  \
    }                                                                    \
    CopyDataFromRules(ctx, BnInOp2Blob, bw_copy_rules_);                 \
  }
OF_PP_FOR_EACH_TUPLE(FLOATING_BOXING_KERNEL_CONCAT_BOX_BACKWARD,
                     FLOATING_DATA_TYPE_SEQ);

#define FLOATING_BOXING_KERNEL_ADD_BOX_FORWARD(type_cpp, type_proto)          \
  template<>                                                                  \
  void BoxingKernel<type_cpp>::AddBoxForward(                                 \
      const KernelCtx& ctx,                                                   \
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {           \
    Blob* out_0 = BnInOp2Blob("out_0");                                       \
    Blob* in_0 = BnInOp2Blob("in_0");                                         \
    Memcpy<DeviceType::kCPU>(ctx.device_ctx, out_0->mut_dptr(), in_0->dptr(), \
                             out_0->ByteSizeOfDataField(),                    \
                             cudaMemcpyKind::cudaMemcpyHostToHost);           \
    for (size_t i = 1; i < op()->input_bns().size(); ++i) {                   \
      Blob* in_i = BnInOp2Blob("in_" + std::to_string(i));                    \
      KernelUtil<DeviceType::kCPU, type_cpp>::BlasAxpy(                       \
          ctx.device_ctx, out_0->shape().elem_cnt(),                          \
          static_cast<type_cpp>(1.0), in_i->dptr<type_cpp>(), 1,              \
          out_0->mut_dptr<type_cpp>(), 1);                                    \
    }                                                                         \
    CopyDataFromRules(ctx, BnInOp2Blob, fw_copy_rules_);                      \
  }
OF_PP_FOR_EACH_TUPLE(FLOATING_BOXING_KERNEL_ADD_BOX_FORWARD,
                     FLOATING_DATA_TYPE_SEQ);

#define NON_FLOATING_BOXING_KERNEL_ADD_BOX_FORWARD(type_cpp, type_proto) \
  template<>                                                             \
  void BoxingKernel<type_cpp>::AddBoxForward(                            \
      const KernelCtx& ctx,                                              \
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {      \
    UNEXPECTED_RUN();                                                    \
  }
OF_PP_FOR_EACH_TUPLE(NON_FLOATING_BOXING_KERNEL_ADD_BOX_FORWARD,
                     INT_DATA_TYPE_SEQ);

#define NON_FLOATING_BOXING_KERNEL_CONCAT_BOX_BACKWARD(type_cpp, type_proto) \
  template<>                                                                 \
  void BoxingKernel<type_cpp>::ConcatBoxBackward(                            \
      const KernelCtx& ctx,                                                  \
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {          \
    UNEXPECTED_RUN();                                                        \
  }
OF_PP_FOR_EACH_TUPLE(NON_FLOATING_BOXING_KERNEL_CONCAT_BOX_BACKWARD,
                     INT_DATA_TYPE_SEQ);

Kernel* CreateBoxingKernel(const OpContext& op_ctx) {
  static const HashMap<int, std::function<Kernel*()>> creators = {
#define BOXING_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new BoxingKernel<type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(BOXING_KERNEL_ENTRY, ALL_DATA_TYPE_SEQ)};
  return creators.at(op_ctx.bn_in_op2data_type().at("in_0"))();
}

COMMAND(AddKernelCreator(OperatorConf::kBoxingConf, CreateBoxingKernel));

}  // namespace oneflow
