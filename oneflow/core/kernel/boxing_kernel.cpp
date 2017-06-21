#include "oneflow/core/kernel/boxing_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/cblas.h"

namespace oneflow {
namespace {
// templates for memory copy 
template<typename device_type> 
void OFMemcpy<device_type>(
    const KernelCtx& ctx, void* dst, 
    const void* src, size_t sz) {
}
void OFMemcpy<DeviceType::kCPU>(
    const KernelCtx& ctx, void* dst, 
    const void* src, size_t sz) {
  memcpy(dst, src, sz); 
}

void OFMemcpy<DeviceType::kGPU>(
    const KernelCtx& ctx, void* dst, 
    const void* src, size_t sz) {
  cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, 
      ctx.device_ctx->cuda_stream());
}

// templates for blobs addition
template<typename device_type, typename floating_point_type>
void OFAddBlob<device_type, floating_point_type>(
    const KernelCtx& ctx, 
    Blob* a, Blob* b) {
}

template<>
void OFAddBlob<DeviceType::kCPU, float>(
    const KernelCtx& ctx, 
    Blob* a, Blob* b) {
  int64_t block_sz = a->shape().elem_cnt(); 
  cblas_saxpy(block_sz, 1.0, 
      static_cast<float*>(a->dptr()), 1, 
      static_cast<float*>(b->mut_dptr()), 1);
}

template<>
void OFAddBlob<DeviceType::kCPU, double>(
    const KernelCtx& ctx, 
    Blob* a, Blob* b) {
  int64_t block_sz = a->shape().elem_cnt(); 
  cblas_daxpy(block_sz, 1.0, 
      static_cast<double*>(a->dptr()), 1, 
      static_cast<double*>(b->mut_dptr()), 1);
}

template<>
void OFAddBlob<DeviceType::kGPU, typename floating_point_type>(
    const KernelCtx& ctx, Blob* a, Blob* b) {
  CHECK_EQ(cublas_axpy<floating_point_type>(
        ctx.device->cublas_handle(),
        a->shape().elem_cnt(), 1.0,
        static_cast<floating_point_type>(a->dptr()), 1,
        static_cast<floating_point_type>(b->mut_dptr()), 1), 
      cudaSuccess);
}

// templates for cblas_axpy
template<typename device_type, typename floating_point_type>
void of_cblas_axpy<device_type, floating_point_type>(
    const KernelCtx& ctx, 
    const int N, const floating_point_type alpha, 
    const floating_point_type* X, const int incX, 
    floating_point_type *Y, const int incY) {
  
}

template<>
void of_cblas_axpy<DeviceType::kCPU, float>(
    const KernelCtx& ctx, 
    const int N, const float alpha, 
    const float *X, const int incX, 
    float *Y, const int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

template<>
void of_cblas_axpy<DeviceType::kCPU, double>(
    const KernelCtx& ctx, 
    const int N, const double alpha, 
    const double *X, const int incX, 
    double *Y, const int incY) {
  cblas_axpy(N, alpha, X, incX, Y, incY);
}

template<typename floating_point_type>
void of_cblas_axpy<DeviceType::kGPU, floating_point_type>(
    const KernelCtx& ctx, 
    const int N, const floating_point_type alpha, 
    const floating_point_type *X, const int incX, 
    floating_point_type *Y, const int incY) {
  CHECK_EQ(cublas_axpy<floating_point_type>(
        ctx.device->cublas_handle(),
        N, alpha, X, incX, Y, incY), 
      cudaSuccess);
}

// templates for cblas_scale
template<typename device_type, typename floating_point_type>
void of_cblas_scal<device_type, floating_point_type>(
    const KernelCtx& ctx,
    const int n, const floating_point_type alpha,
    floating_point_type* x, int incx) {

}

template<> 
void of_cblas_scal<DeviceType::kCPU, double>(
    const KernelCtx& ctx,
    const int n, const double alpha,
    double* x, int incx) {
  cblas_dscal(n, alpha, x, incx);
}

template<> 
void of_cblas_scal<DeviceType::kCPU, float>(
    const KernelCtx& ctx,
    const int n, const float alpha,
    float* x, int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template<typename floating_point_type> 
void of_cblas_scal<DeviceType::kGPU, floating_point_type>(
    const KernelCtx& ctx, 
    const int n, const floating_point_type alpha,
    floating_point_type* x, int incx) {
  CHECK_EQ(cublas_cublas<floating_point_type>(
        ctx.device->cublas_handle(),
        n, alpha, x, incx), 
      cudaSuccess);
}

template<DeviceType::kGPU

} // namespace 

template<typename floating_point_type> 
void BoxingKernel<DeviceType::kCPU, floating_point_type>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    fw_func_ = &BoxingKernel<device_type, floating_point_type>::ConcatBoxForward;
    bw_func_ = &BoxingKernel<device_type, floating_point_type>::ConcatBoxBackward;
  } else {
    fw_func_ = &BoxingKernel<device_type, floating_point_type>::AddBoxForward;
    bw_func_ = &BoxingKernel<device_type, floating_point_type>::AddBoxBackward;
  }
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::InferCopyRules(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) {
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    // Infer fw copy rules
    InferCopyRulesFromBns(BnInOp2BlobPtr, op()->input_bns(), 
        op()->output_bns(), fw_copy_rules); 
    // Infer bw copy rules BUG HERE!!!!
    InferCopyRulesFromBns(BnInOp2BlobPtr, op()->output_diff_bns(),
        op()->input_diff_bns(), bw_copy_rules); 
  } 
  
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    std::vector<string>& src_bns,
    std::vector<string>& dst_bns,
    std::vector<copy_rule>& copy_rules) {
  std::vector<const Shape&> src_shapes, dst_shapes;
  std::vector<const string&> bns;
  for (const std::string& bn: src_bns) {
    src_shapes.push_back(BnInOp2BlobPtr(bn)->shape());
    bns.push_back(bn); 
  }
  for (const std::string& bn: dst_bns) {
    dst_shapes.push_back(BnInOp2BlobPtr(bn)->shape()):
    bns.push_back(bn);
  }

  int concat_axis = op()->op_conf().boxing_conf().concat_box().axis();
  // calculate the single copy-block size
  uint64_t block_sz_1 = 1;
  vector<int64_t>& dim_vec = src_shapes.dim_vec();
  for (int i=2; i<dim_vec.size(); ++i) {
      block_sz_1 *= dim_vec.at(i);
  }
  uint64_t block_sz_0 = block_sz * dim_vec.at(1);
  ConstructRulesFromShape(
      id2bn, src_shapes, dst_shapes, 
      block_sz_0, block_sz_1, concat_axis,
      copy_rules);
  
}

// Construct direct copy rules between blobs from blob shapes, results are
// saved into rules vector
template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::ConstructRulesFromShape(
    std::vector<const string&>& bns,
    std::vector<const Shape&>& in_shapes,
    std::vector<const Shape&>& out_shapes,
    uint64_t block_sz, uint64_t block_sz_0, int concat_axis,
    std::vector<struct copy_rules>& rules) {
  int in_idx = 0; 
  uint64_t in_offset = 0;
  uint64_t in_cap = in_shapes[0].At(concat_axis);
  for (int out_idx=0; out_idx<out_shapes.size(); ++out_idx) {
    uint64_t out_cap = out_shapes[out_idx].At(concat_axis);
    uint64_t out_offset = 0;
    while (out_offset < out_cap) {
      uint64_t p = min(in_cap-in_offset, out_cap-out_offset);
      uint64_t seg_cnt = (concat_axis==0) ? 1 : in_shapes[0].At(0);
      for(int i=0; i<seg_cnt; ++i) {
        struct copy_rule re;
        re.src_bn = bns.at(in_idx);
        re.dst_bn = bns.at(out_idx + in_shapes.size());
        re.src_offset = in_offset * block_sz + i * block_sz_0;
        re.dst_offset = out_offset * block_sz + i * block_sz_0;
        re.copy_sz = p * block_sz; 
        rules.push_back(std::move(re));
      }
      out_offset += p;
      in_offset += p;
      if (in_offset == in_cap) {
        in_cap = 0;
        if (++in_idx == in_shapes.size()) break;
      }
    }
    if (in_idx == in_shapes.size()) break;
  }
  
  const BoxingOpConf& boxing_conf = op()->op_conf().boxing_conf();
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    auto bn_iter = bn.begin() + in_shapes.size();
    const string& clone_base_bn = *bn_iter++;
    uint64_t blob_sz = block_sz_0 * out_shape.front().At(0);
    while(bn_iter != bns.end()) {
      struct copy_rule re;     
      re.src_bn = clone_base_bn;
      re.dst_bn = *bn_iter;
      re.src_offset = re.dst_offset = 0;
      re.copy_sz = blob_sz;
      rules.push_back(std::move(re));
      ++bn_iter;
    }
  }
}
template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  *fw_func(ctx, BnInOp2BlobPtr); 
}


template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  *bw_func(ctx, BnInOp2BlobPtr);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::ConcatBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  if (fw_copy_rules.empty()) {
    InferCopyRules(BnInOp2BlobPtr);
  }
  
  std::function<void()> fp = [BnInOp2BlobPtr, &fw_copy_rules]() {
    for (auto rule : fw_copy_rules) {
      Blob* src_blob = BnInOp2BlobPtr(rule.src_bn);
      Blob* dst_blob = BnInOp2BlobPtr(rule.dst_bn);
      OFMemcpy<device_type>(ctx, dst_blob->mut_dptr() + rule.dst_offest, \
         src_blob->dptr() + rule.src_offset, \
         rule.copy_sz);
    }
  }
  ctx.AddCallBack(fp);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::ConcatBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  if (bw_copy_rules.empty()) {
    InferCopyRules(BnInOp2BlobPtr);
  }

  std::function<void()> fp = [BnInOp2BlobPtr, &bw_copy_rules]() {
    for (auto rule : bw_copy_rules) {
      Blob* src_blob = BnInOp2BlobPtr(rule.src_bn);
      Blob* dst_blob = BnInOp2BlobPtr(rule.dst_bn);
      OFMemcpy(ctx, dst_blob->mut_dptr() + rule.dst_offest, \
         src_blob->dptr() + rule.src_offset, \
         rule.copy_sz);
    }
  };
  ctx.AddCallBack(fp);
}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::AddBoxForward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  
  // 1. Copy the first input blob to first dst blob;
  Blob* dst_blob = BnInOp2BlobPtr(op()->output_bns().front());
  Blob* fst_in_blob = op()->input_bns().front();
  std::function<void()> fp1 = [dst_blob, fst_in_blob]() {
    OFMemcpy(ctx, 
        dst_blob->mut_dptr(), fst_in_blob->dptr(),
        dst_blob->shape().elem_cnt() * sizeof(floating_point_type));
  };
  ctx.AddCallBack(fp1);
      
  // 2. add all the remaining input blob to first blob;
  vector<string>& input_bns = op()->input_bns(); 
  for (size_t i=1; i<input_bns.size(); ++i) {
    OFAddBlob(ctx, BnInOp2BlobPtr(input_bns.at(i)), dst_blob);
  }

  // 3. If output box is in clone mode, copy it to remaining boxes.
  auto boxing_conf = op()->op_conf()->boxing_conf();
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    vector<string>& output_bns = op()->output_bns();
    if (output_bns.size() <= 1) 
      return;
    std::function<void()> fp2 = [input_bns, output_bns]() {
      for (size_t i=1; i<input_bns.size(); ++i) {
        OFMemcpy(ctx, 
            BnInOp2BlobPtr(input_bns.at(i))->mut_dptr(),
            dst_blob->dptr(),
            dst_blob->shape().elem_cnt() * sizeof(floating_point_type));
      }
    };
    ctx.AddCallBack(fp2);
  }

}

template<typename floating_point_type>
void BoxingKernel<DeviceType::kCPU, floating_point_type>::AddBoxBackward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* out_blob = BnInOp2BlobPtr(op()->output_diff_bns().front());
  vector<string>& idbns = op()->input_diff_bns();
  Blob* fst_in_blob = BnInOp2BlobPtr(idbns.front());
  size_t in_dbns_sz = idbns.size();

  std::function<void()> fp1 = [out_blob, fst_in_blob, \
                              idbns, in_dbns_sz]() {
    // 1. scale out_dbn by 1.0/input_number to fst in_dbn;
    uint64_t bsz = fst_in_blob->shape().elem_cnt();
    of_cblas_scal(ctx, bsz, 0.0, fst_in_blob->mut_dptr(), 1);
    of_cblas_axpy(
        ctx, bsz, 1.0/in_dbns_sz, 
        out_blob->dptr(), 1,
        fst_in_blob->mut_dptr(), 1
        );

    // 2. copy in_dbn to all other input diff blobs;
    for (size_t i=1; i<idbns.size(); ++i) {
      Blob* dst_blob = BnInOp2BlobPtr(idbns.at(i));
      OFMemcpy(
          dst_blob->mut_dptr(), fst_in_blob->dptr(),
          fst_in_blob->shape().elem_cnt() * \
          sizeof(floating_point_type)
          );
    }
  };
  ctx.AddCallBack(fp1);

}
