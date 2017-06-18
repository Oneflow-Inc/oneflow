#include "oneflow/core/kernel/clone_kernel.h"
#include <string>
#include <cstring>

namespace oneflow {

namespace {
template<typename T>
cublasStatus_t cublas_axpy(const int N,
                           const T *alpha,
                           const T *x, int incx,
                                 T *y, int incy){
//LOG
}

template<>
cublasStatus_t cublas_axpy<float>(const int N,
                                  const float *alpha,
                                  const float *x, int incx,
                                        float *y, int incy){
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template<>
cublasStatus_t cublas_axpy<double>(const int N,
                                   const double *alpha,
                                   const double *x, int incx,
                                         double *y, int incy){
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

}// namespace

template<typename floating_point_type>
void CloneKernel<DeviceType::kCPU, floating_point_type>::Forward(
  const KernelContext&,
  std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  Blob* in_blob = bn_in_op2blob_ptr(op()->SoleIbn());
  const vector<std::string>& obns = op()->output_bns();
  for(auto& obn : obns) {
    Blob* out_blob = bn_in_op2blob_ptr(obn);
    memcpy(out_blob->mut_dptr(), in_blob->dptr(),
           in_blob->shape().elem_cnt()*sizeof(floating_point_type));
  }
}

template<typename floating_point_type>
void CloneKernel<DeviceType::kCPU, floating_point_type>::Backward(
    const KernelContext&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  Blob* in_blob = bn_in_op2blob_ptr(op()->SoleIbn());
  memset(in_blob->dptr(), 0, in_blob->shape().elem_cnt()*sizeof(floating_point_type));
  const vector<std::string>& obns = op()->output_bns();
  const floating_point_type* alpha = new floating_point_type(1.0);
  for(auto& obn : obns) {
    Blob* out_blob = bn_in_op2blob_ptr(obn);
    //async?
    cblas_axpy<floating_point_type>(in_blob->shape().elem_cnt(), alpha,
                                    static_cast<floating_point_type*>(out_blob->dptr()), 1,
                                    static_cast<floating_point_type*>(in_blob->mut_dptr()), 1);
    /*
    for(int i=0;i!=in_blob->shape().elem_cnt();++i){
      in_dptr[i] += out_dptr[i];
    }
    */
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(CloneKernel);
REGISTER_KERNEL(OperatorConf::kCloneConf, CloneKernel);

}  // namespace oneflow
