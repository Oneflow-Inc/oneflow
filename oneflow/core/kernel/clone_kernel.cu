#include "oneflow/core/kernel/clone_kernel.h"
#include <string>
#include <typeinfo>

namespace oneflow {

namespace {

template<typename T>
cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const T *alpha,
                           const T *x, int incx,
                           T *y, int incy){
  LOG(FATAL) << "floating_point_type should be flaot or double";
}

template<>
cublasStatus_t cublas_axpy<float>(cublasHandle_t handle, int n,
                                  const float *alpha,
                                  const float *x, int incx,
                                  float *y, int incy){
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template<>
cublasStatus_t cublas_axpy<double>(cublasHandle_t handle, int n,
                                   const double *alpha,
                                   const double *x, int incx,
                                   double *y, int incy){
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

} // namespace

template<typename floating_point_type>
void CloneKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in_blob = BnInOp2BlobPtr(op()->SoleIbn());
  for(const std::string& obn : op()->output_bns()) {
    Blob* out_blob = BnInOp2BlobPtr(obn);
    CHECK_EQ(cudaMemcpyAsync(out_blob->mut_dptr(),
                             in_blob->dptr(),
                             in_blob->shape().elem_cnt() * sizeof(floating_point_type),
                             cudaMemcpyDeviceToDevice,
                             ctx.device_ctx->cuda_stream()),
             cudaSuccess);
  }
}

template<typename floating_point_type>
void CloneKernel<DeviceType::kGPU, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* idbn_blob = BnInOp2BlobPtr(op()->SoleIdbn());
  const std::vector<std::string>& odbns = op()->output_diff_bns();
  if (odbns.size() == 0) return;
  CHECK_EQ(cudaMemcpyAsync(idbn_blob->mut_dptr(),
                           BnInOp2BlobPtr(odbns[0])->dptr(),
                           idbn_blob->shape().elem_cnt() * sizeof(floating_point_type),
                           cudaMemcpyDeviceToDevice,
                           ctx.device_ctx->cuda_stream()),
           cudaSuccess);
  const floating_point_type alpha = {1.0f};
  for(size_t i = 1; i != odbns.size(); ++i) {
    const Blob* odbn_blob = BnInOp2BlobPtr(odbns[i]);
    CHECK_EQ(cublas_axpy<floating_point_type>(
                 ctx.device_ctx->cublas_handle(),
                 idbn_blob->shape().elem_cnt(), &alpha,
                 static_cast<const floating_point_type*>(odbn_blob->dptr()), 1,
                 static_cast<floating_point_type*>(idbn_blob->mut_dptr()), 1),
             cudaSuccess);
  }
}

INSTANTIATE_GPU_KERNEL_CLASS(CloneKernel);
REGISTER_GPU_KERNEL(OperatorConf::kCloneConf, CloneKernel);

}  // namespace oneflow
