#include "oneflow/core/kernel/clone_kernel.h"
#include <string>
#include <cstring>
#include "oneflow/core/common/cblas.h"

namespace oneflow {

namespace {
template<typename T>
void cblas_axpy(const int N,
                const T alpha,
                const T *x, const int incx,
                T *y, const int incy) {
  LOG(FATAL) << "floating_point_type should be float or dounle";
}

template<>
void cblas_axpy<float>(const int N,
                       const float alpha,
                       const float *x, const int incx,
                       float *y, const int incy) {
  cblas_saxpy(N, alpha, x, incx, y, incy);
}

template<>
void cblas_axpy<double>(const int N,
                        const double alpha,
                        const double *x, const int incx,
                        double *y, const int incy) {
  cblas_daxpy(N, alpha, x, incx, y, incy);
}

}// namespace

template<typename floating_point_type>
void CloneKernel<DeviceType::kCPU, floating_point_type>::Forward(
  const KernelCtx& ctx,
  std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob = BnInOp2BlobPtr(op()->SoleIbn());
  const std::vector<std::string>& obns = op()->output_bns();
  for(auto& obn : obns) {
    Blob* out_blob = BnInOp2BlobPtr(obn);
    auto cpu_stream_memcpy = [&]() {
      memcpy(out_blob->mut_dptr(),
             in_blob->dptr(),
             in_blob->shape().elem_cnt()*sizeof(floating_point_type));
    };
    ctx.device_ctx->cpu_stream()->Send(cpu_stream_memcpy);
  }
}

template<typename floating_point_type>
void CloneKernel<DeviceType::kCPU, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* idbn_blob = BnInOp2BlobPtr(op()->SoleIdbn());
  const std::vector<std::string>& odbns = op()->output_diff_bns();
  if (odbns.size() == 0) return;
  auto cpu_stream_memcpy = [=]() {
    memcpy(idbn_blob->mut_dptr(),
           BnInOp2BlobPtr(odbns[0])->dptr(),
           idbn_blob->shape().elem_cnt() * sizeof(floating_point_type));
  };
  ctx.device_ctx->cpu_stream()->Send(cpu_stream_memcpy);
  for(int i = 1; i != odbns.size(); ++i) {
    Blob* out_blob = BnInOp2BlobPtr(odbns[i]);
    auto cpu_stream_axpy = [=]() {
      cblas_axpy<floating_point_type>(idbn_blob->shape().elem_cnt(), 1.0,
                                      reinterpret_cast<const floating_point_type*>(out_blob->dptr()), 1,
                                      reinterpret_cast<floating_point_type*>(idbn_blob->mut_dptr()), 1);
    };
    ctx.device_ctx->cpu_stream()->Send(cpu_stream_axpy);
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(CloneKernel);
REGISTER_CPU_KERNEL(OperatorConf::kCloneConf, CloneKernel);

}  // namespace oneflow
