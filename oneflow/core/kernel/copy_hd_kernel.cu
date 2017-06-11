#include "oneflow/core/kernel/copy_hd_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelContext& ctx,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  const std::string& ibn = op()->SoleIbn();
  Blob* in_blob = bn_in_op2blob_ptr(ibn);
  const std::string& obn = op()->SoleObn();
  Blob* out_blob = bn_in_op2blob_ptr(obn);
  
  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();
  
  if (copy_hd_conf.type() == CopyHdOpConf_Type_H2D) {
    cudaMemcpyAsync(out_blob->mut_dptr(), in_blob->dptr(), 
                    in_blob->shape().elem_cnt()*sizeof(floating_point_type),
                    cudaMemcpyHostToDevice,
                    *(ctx.cuda_stream));
  } else {
    cudaMemcpyAsync(out_blob->mut_dptr(), in_blob->dptr(), 
                    in_blob->shape().elem_cnt()*sizeof(floating_point_type),
                    cudaMemcpyDeviceToHost, 
                    *(ctx.cuda_stream));
  }
}

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::Backward(
    const KernelContext& ctx,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  const std::string& ibn = op()->SoleIbn();
  Blob* in_blob = bn_in_op2blob_ptr(ibn);
  const std::string& obn = op()->SoleObn();
  Blob* out_blob = bn_in_op2blob_ptr(obn);

  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == CopyHdOpConf_Type_H2D) {
    cudaMemcpyAsync(out_blob->mut_dptr(), in_blob->dptr(), 
                    in_blob->shape().elem_cnt()*sizeof(floating_point_type),
                    cudaMemcpyDeviceToHost,
                    *(ctx.cuda_stream));
  } else {
    cudaMemcpyAsync(out_blob->mut_dptr(), in_blob->dptr(), 
                    in_blob->shape().elem_cnt()*sizeof(floating_point_type),
                    cudaMemcpyHostToDevice,
                    *(ctx.cuda_stream));
  }
}

INSTANTIATE_GPU_KERNEL_CLASS(CopyHdKernel);

}  // namespace oneflow
