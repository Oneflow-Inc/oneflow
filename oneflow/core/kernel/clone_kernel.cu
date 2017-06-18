#include "oneflow/core/kernel/clone_kernel.h"
#include <string>
#include <typeinfo>

namespace {

template<typename T>
cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                           const T *alpha,
                           const T *x, int incx,
                           const T *y, int incy){
//LOG
}

template<>
cublasStatus_t cublas_axpy<float>(cublasHandle_t handle, int n,
                                  const float *alpha,
                                  const float *x, int incx,
                                  const float *y, int incy){
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template<>
cublasStatus_t cublas_axpy<double>(cublasHandle_t handle, int n,
                                   const double *alpha,
                                   const double *x, int incx,
                                   const double *y, int incy){
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

} // namespace


namespace oneflow {

template<typename floating_point_type>
void CloneKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelContext& ctx,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  Blob* in_blob = bn_in_op2blob_ptr(op()->SoleIbn());
  const vector<std::string>& obns = op()->output_bns();
  for(auto& obn : obns) {
      Blob* out_blob = bn_in_op2blob_ptr(obn);
      CHECK_EQ(cudaMemcpyAsync(out_blob->mut_dptr(),
                               in_blob->dptr(),
                               in_blob->shape().elem_cnt() * sizeof(floating_point_type),
                               cudaMemcpyDeviceToDevice,
                               ctx.device_ctx->cuda_stream),
               cudaSuccess);
  }
}

template<typename floating_point_type>
void CloneKernel<DeviceType::kGPU, floating_point_type>::Backward(
    const KernelContext&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  Blob* in_blob = bn_in_op2blob_ptr(op()->SoleIbn());
  //const int blob_mem_size = in_blob->shape().elem_cnt() * sizeof(floating_point_type);
  //floating_point_type* in_blob_device_ptr;
  //CHECK_EQ(cudaMalloc(in_blob_device_ptr, blob_mem_size), cudaSuccess);
  CHECK_EQ(cudaMemsetAsync(in_blob_device_ptr, 0,
                           in_blob->shape().elem_cnt() * sizeof(floating_point_type),
                           ctx.device_ctx->cuda_stream),
           cudaSuccess);
  const vector<std::string>& obns = op()->output_bns();
  const floating_point_type* alpha = new floating_point_type(1.0);
  for(auto& obn : obns) {
    Blob* out_blob = bn_in_op2blob_ptr(obn);
    //floating_point_type* out_blob_device_ptr;
    //CHECK_EQ(cudaMalloc(out_blob_device_ptr, blob_mem_size),
    //         cudaSuccess);
    //CHECK_EQ(cudaMemcpyAsync(out_blob_device_ptr,
    //                         static_cast<floating_point_type>(out_blob->dptr()),
    //                         blob_mem_size,
    //                         ctx.device_ctx->cuda_stream),
    //         cudaSuccess);
    CHECK_EQ(cublas_axpy<floating_point_type>(ctx.device_ctx->cublas_handle, alpha,
                                              in_blob->shape().elem_cnt(),
                                              static_cast<floating_point_type*>(out_blob->dptr()), 1,
                                              static_cast<floating_point_type*>(in_blob->mut_dptr()), 1),
             cudaSuccess);
    //CHECK_EQ(cudaFree(out_blob_device_ptr), cudaSuccess);
  }
  delete alpha;
  //CHECK_EQ(cudaMemcpyAsync(in_blob->mut_dptr(),
  //                         in_blob_device_ptr,
  //                         blob_mem_size,
  //                         cudaMemcpyDeviceToHost,
  //                         ctx.device_ctx->cuda_stream),
  //         cudaSuccess);
  //CHECK_EQ(cudaFree(in_blob_device_ptr), cudaSuccess);
}

INSTANTIATE_GPU_KERNEL_CLASS(CloneKernel);

}  // namespace oneflow
