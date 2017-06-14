#include "oneflow/core/kernel/copy_hd_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

void CopyH2DAsync(Blob* in_blob, Blob* out_blob,
                  const cudaStream_t& cuda_stream, const size_t type_size) {
  CHECK_EQ(cudaMemcpyAsync(out_blob->mut_dptr(),
                           in_blob->dptr(),
                           in_blob->shape().elem_cnt() * type_size,
                           cudaMemcpyHostToDevice,
                           cuda_stream),
           cudaSuccess);
}

void CopyD2HAsync(Blob* in_blob, Blob* out_blob,
                  const cudaStream_t& cuda_stream, const size_t type_size) {
  CHECK_EQ(cudaMemcpyAsync(out_blob->mut_dptr(),
                           in_blob->dptr(),
                           in_blob->shape().elem_cnt() * type_size,
                           cudaMemcpyDeviceToHost,
                           cuda_stream),
           cudaSuccess);
}

}  // namespac

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);
  
  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
    ForwardCopy = CopyH2DAsync;
    BackwardCopy = CopyD2HAsync;
  } else {
    ForwardCopy = CopyD2HAsync;
    BackwardCopy = CopyH2DAsync;
  }
}

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob  = BnInOp2BlobPtr( op()->SoleIbn() );
  Blob* out_blob = BnInOp2BlobPtr( op()->SoleObn() );

  (*ForwardCopy)(in_blob, out_blob,
                 ctx.cuda_stream(), sizeof(floating_point_type));
}

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob  = BnInOp2BlobPtr( op()->SoleOdbn() );
  Blob* out_blob = BnInOp2BlobPtr( op()->SoleIdbn() );

  (*BackwardCopy)(in_blob, out_blob,
                 ctx.cuda_stream(), sizeof(floating_point_type));
}

INSTANTIATE_GPU_KERNEL_CLASS(CopyHdKernel);
REGISTER_GPU_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);

}  // namespace oneflow
