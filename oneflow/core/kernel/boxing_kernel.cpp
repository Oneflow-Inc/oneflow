#include "oneflow/core/kernel/boxing_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/blas/cblas.h"

namespace oneflow {

// templates for memory copy 
template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::OFMemcpy(
    const KernelCtx& ctx, void* dst, 
    const void* src, size_t sz) {
  memcpy(dst, src, sz); 
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::OFBlobCpy(
    const KernelCtx& ctx, 
    const Blob* a, Blob* b) {
  memcpy( static_cast<floating_point_type*>(b->mut_dptr()), 
      static_cast<const floating_point_type*>(a->dptr()),
      sizeof(floating_point_type) * a->shape().elem_cnt());
}

// templates for blobs addition
template<>
void BoxingKernel<DeviceType::kCPU, float>::OFBlobAdd(
    const KernelCtx& ctx, 
    const Blob* a, Blob* b) {
  int64_t block_sz = a->shape().elem_cnt(); 
  cblas_saxpy(block_sz, 1.0, 
      static_cast<const float*>(a->dptr()), 1, 
      static_cast<float*>(b->mut_dptr()), 1);
}

template<>
void BoxingKernel<DeviceType::kCPU, double>::OFBlobAdd(
    const KernelCtx& ctx,
    const Blob* a, Blob* b) {
  int64_t block_sz = a->shape().elem_cnt(); 
  cblas_daxpy(block_sz, 1.0, 
      static_cast<const double*>(a->dptr()), 1, 
      static_cast<double*>(b->mut_dptr()), 1);
}

// templates for cblas_axpy
template<>
void BoxingKernel<DeviceType::kCPU, float>::OFBlasAxpy(
    const KernelCtx& ctx, 
    const int N, const float alpha, 
    const float *X, const int incX, 
    float *Y, const int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

template<>
void BoxingKernel<DeviceType::kCPU, double>::OFBlasAxpy(
    const KernelCtx& ctx, 
    const int N, const double alpha, 
    const double *X, const int incX, 
    double *Y, const int incY) {
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

// templates for cblas_scale
template<> 
void BoxingKernel<DeviceType::kCPU, double>::OFBlasScal(
    const KernelCtx& ctx,
    const int n, const double alpha,
    double* x, int incx) {
  cblas_dscal(n, alpha, x, incx);
}

template<> 
void BoxingKernel<DeviceType::kCPU, float>::OFBlasScal(
    const KernelCtx& ctx,
    const int n, const float alpha,
    float* x, int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template<typename floating_point_type> 
void BoxingKernel<DeviceType::kALL, floating_point_type>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    fw_func_ = &BoxingKernel<DeviceType::kALL, \
               floating_point_type>::ConcatBoxForward;
    bw_func_ = &BoxingKernel<DeviceType::kALL, \
               floating_point_type>::ConcatBoxBackward;
  } else {
    fw_func_ = &BoxingKernel<DeviceType::kALL, \
               floating_point_type>::AddBoxForward;
    bw_func_ = &BoxingKernel<DeviceType::kALL, \
               floating_point_type>::AddBoxBackward;
  }
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::InferCopyRules(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
    // This box kernel MUST be concat ==> (split/clone) box kernel
    // Infer fw copy rules
    InferCopyRulesFromBns(BnInOp2BlobPtr, op()->input_bns(), 
        op()->output_bns(), fw_copy_rules); 
    auto boxing_conf = op()->op_conf().boxing_conf();
    // Add blob copy rules for clone-box
    if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      ConstructFwCloneRules(BnInOp2BlobPtr);
    }

    // Infer bw copy rules
    InferCopyRulesFromBns(BnInOp2BlobPtr, op()->input_diff_bns(),
        op()->output_diff_bns(), bw_copy_rules); 

    // Reverse back input && output diff blob for each backward rule
    for (auto& rule : bw_copy_rules) {
      std::swap(rule.src_bn, rule.dst_bn);
      std::swap(rule.src_offset, rule.dst_offset);
    }
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns,
    std::vector<copy_rule>& copy_rules) const {

  // P.S This routine will be called only once, thus some performance
  // loss seems ok.
  std::map<const std::string*, int64_t> src_bn2slice, dst_bn2slice;
  int concat_axis = op()->op_conf().boxing_conf().concat_box().axis();
  for (const std::string& bn : src_bns) {
    src_bn2slice.insert(make_pair(&bn, \
          (BnInOp2BlobPtr(bn)->shape().At(concat_axis))));
  }
  for (const std::string& bn : dst_bns) {
    dst_bn2slice.insert(make_pair(&bn, \
          (BnInOp2BlobPtr(bn)->shape().At(concat_axis))));
  }
  
  Blob* src_fst_blob = BnInOp2BlobPtr(src_bns.front());
  int64_t slice_sz = src_fst_blob->shape().Count(2);
  int64_t seg_cnt = (concat_axis==0) ? 1 : (src_fst_blob->shape().At(0));

  ConstructCopyRulesFromShape(
      src_bn2slice, dst_bn2slice, seg_cnt, 
      slice_sz, concat_axis, copy_rules
      );
}

// Construct direct copy rules between blobs from blob shapes, results are
template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, \
       floating_point_type>::ConstructCopyRulesFromShape (
  std::map<const std::string*, int64_t>& src_bn2slice, 
  std::map<const std::string*, int64_t>& dst_bn2slice,
  int64_t seg_cnt, int64_t slice_sz, int concat_axis, 
  std::vector<struct copy_rule>& rules) const {

  auto src_iter = src_bn2slice.begin();
  auto dst_iter = dst_bn2slice.begin();
  int64_t src_offset = 0, src_cap = src_iter->second;
  int64_t dst_offset = 0, dst_cap = dst_iter->second;
  while (src_iter != src_bn2slice.end() && \
      dst_iter != dst_bn2slice.end()) {
    dst_offset = 0, dst_cap = dst_iter->second;
    while (dst_offset < dst_cap) {
      int64_t p = std::min(src_cap-src_offset, dst_cap-dst_offset);
      for (size_t i=0; i<seg_cnt; ++i) {
        struct copy_rule cr;
        cr.src_bn = *src_iter->first;
        cr.dst_bn = *src_iter->first;
        cr.src_offset = (src_offset + i * src_cap) * slice_sz;
        cr.dst_offset = (dst_offset + i * dst_cap) * slice_sz;
        cr.copy_sz = p * slice_sz * sizeof(floating_point_type);
        rules.push_back(std::move(cr));
      }
      src_offset += p, dst_offset += p;
      if (src_offset == src_cap) {
        src_cap = 0;
        if (++src_iter == src_bn2slice.end()) break;
      }
    } // while current dst_cap is not full
    ++dst_iter;
  }
  ASSERT_EQ(dst_iter, dst_bn2slice.end());
  ASSERT_EQ(dst_offset, dst_cap);
}

// If out box case is a clone box, then previous rules would only encapsulate 
// the input source boxes to the first single output box. Hence, we have to 
// clone the first output box to the remaining output boxes.
template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::ConstructFwCloneRules(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::vector<std::string>& obns = op()->output_bns();
  int64_t copy_sz = sizeof(floating_point_type) * \
                    BnInOp2BlobPtr(obns.front())->shape().elem_cnt();
  for (size_t i=1; i<obns.size(); ++i) {
    struct copy_rule cr;
    cr.src_bn = obns.front();
    cr.dst_bn = obns.at(i);
    cr.src_offset = 0;
    cr.dst_offset = 0;
    cr.copy_sz = copy_sz;
    fw_copy_rules.push_back(std::move(cr));
  }
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  (this->*fw_func_)(ctx, BnInOp2BlobPtr); 
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  (this->*bw_func_)(ctx, BnInOp2BlobPtr);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::ConcatBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  // 1. Infer copy-rules at first time
  if (fw_copy_rules.empty()) {
    InferCopyRules(BnInOp2BlobPtr);
  }
  
  std::function<void()> fp = [BnInOp2BlobPtr, this, ctx]() {
    for (auto rule : this->fw_copy_rules) {
      Blob* src_blob = BnInOp2BlobPtr(rule.src_bn);
      Blob* dst_blob = BnInOp2BlobPtr(rule.dst_bn);
      OFMemcpy(ctx, static_cast<floating_point_type*>(dst_blob->mut_dptr()) \
          + rule.dst_offset, \
         static_cast<const floating_point_type*>(src_blob->dptr()) \
         + rule.src_offset, rule.copy_sz);
    }
  };
  ctx.device_ctx->AddCallBack(fp);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::ConcatBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  // 1. Infer copy-rules at first time
  if (bw_copy_rules.empty()) {
    InferCopyRules(BnInOp2BlobPtr);
  // However, this should never happend at backward routine
    UNEXPECTED_RUN();
  }

  std::function<void()> fp = [BnInOp2BlobPtr, this, ctx]() {
    for (auto rule : this->bw_copy_rules) {
      Blob* src_blob = BnInOp2BlobPtr(rule.src_bn);
      Blob* dst_blob = BnInOp2BlobPtr(rule.dst_bn);
      OFMemcpy(ctx, static_cast<floating_point_type*>(dst_blob->mut_dptr()) \
          + rule.dst_offset, \
         static_cast<const floating_point_type*>(src_blob->dptr()) \
         + rule.src_offset, rule.copy_sz);
    }
  };

  ctx.device_ctx->AddCallBack(fp);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::AddBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {

  Blob* dst_blob = BnInOp2BlobPtr(op()->output_bns().front());
  Blob* fst_in_blob = BnInOp2BlobPtr(op()->input_bns().front());

  std::function<void()> fp1 = [dst_blob, fst_in_blob, ctx, \
                              BnInOp2BlobPtr, this]() {
  // 1. Copy the first input blob to first dst blob;
    OFBlobCpy(ctx, fst_in_blob, dst_blob);

  // 2. Add all the remaining input blob to first dst blob;
    const std::vector<std::string>& input_bns = this->op()->input_bns(); 
    for (size_t i=1; i<input_bns.size(); ++i) {
      OFBlobAdd(ctx, BnInOp2BlobPtr(input_bns.at(i)), dst_blob);
    }
  };
  ctx.device_ctx->AddCallBack(fp1);
      
  // 3. If output box is in clone mode, copy it to remaining boxes.
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    const std::vector<std::string>& output_bns = op()->output_bns();
    if (output_bns.size() <= 1) 
      return;
    std::function<void()> fp2 = [output_bns, dst_blob, \
                                ctx, BnInOp2BlobPtr, this]() {
      for (size_t i=1; i<output_bns.size(); ++i) {
        OFBlobCpy(ctx, dst_blob, BnInOp2BlobPtr(output_bns.at(i)));
      }
    };
    ctx.device_ctx->AddCallBack(fp2);
  }

}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kALL, floating_point_type>::AddBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* out_diff = BnInOp2BlobPtr(op()->output_diff_bns().front());
  const std::vector<std::string>& idbns = op()->input_diff_bns();
  Blob* fst_in_diff = BnInOp2BlobPtr(idbns.front());

  std::function<void()> fp = [out_diff, fst_in_diff, idbns, ctx, \
                              BnInOp2BlobPtr, this]() {
  // 1. scale out_dbn by 1.0/input_number to fst idbn;
    uint64_t bsz = fst_in_diff->shape().elem_cnt();
    size_t idbns_sz = idbns.size();
    OFBlasScal(ctx, bsz, 0.0, \
        static_cast<floating_point_type*>(fst_in_diff->mut_dptr()), 1);
    OFBlasAxpy(
        ctx, bsz, 1.0/idbns_sz, 
        static_cast<const floating_point_type*>(out_diff->dptr()), 1,
        static_cast<floating_point_type*>(fst_in_diff->mut_dptr()), 1
        );

  // 2. copy in_dbn to all other input diff blobs;
    for (size_t i=1; i<idbns_sz; ++i) {
      OFBlobCpy(ctx, fst_in_diff, BnInOp2BlobPtr(idbns.at(i)));
    }
  };

  ctx.device_ctx->AddCallBack(fp);
}

INSTANTIATE_CPU_KERNEL_CLASS(BoxingKernel);
REGISTER_CPU_KERNEL(OperatorConf::kBoxingConf, BoxingKernel);

} // namespace oneflow
