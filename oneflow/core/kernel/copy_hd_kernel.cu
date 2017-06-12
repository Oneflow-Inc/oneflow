#include "oneflow/core/kernel/copy_hd_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

void CopyH2DAsync(Blob* in_blob, Blob* out_blob,
                  const KernelContext& ctx, size_t type_size) {
  CHECK_EQ(cudaMemcpyAsync(out_blob->mut_dptr(),
                           in_blob->dptr(),
                           in_blob->shape().elem_cnt() * type_size,
                           cudaMemcpyHostToDevice,
                           *(ctx.cuda_stream)),
           cudaSuccess);
}

void CopyD2HAsync(Blob* in_blob, Blob* out_blob,
                  const KernelContext& ctx, size_t type_size) {
  CHECK_EQ(cudaMemcpyAsync(out_blob->mut_dptr(),
                           in_blob->dptr(),
                           in_blob->shape().elem_cnt() * type_size,
                           cudaMemcpyHostToDevice,
                           *(ctx.cuda_stream)),
           cudaSuccess);
}

}  // namespace

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelContext& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::string& ibn = op()->SoleIbn();
  Blob* in_blob = BnInOp2BlobPtr(ibn);
  const std::string& obn = op()->SoleObn();
  Blob* out_blob = BnInOp2BlobPtr(obn);
  size_t type_size = sizeof(floating_point_type);

  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == copy_hd_conf.H2D) {
    CopyH2DAsync(in_blob, out_blob, ctx, type_size);
  } else {
    CopyD2HAsync(in_blob, out_blob, ctx, type_size);
  }
}

template<typename floating_point_type>
void CopyHdKernel<DeviceType::kGPU, floating_point_type>::Backward(
    const KernelContext& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::string& odbn = op()->SoleOdbn();
  Blob* in_blob = BnInOp2BlobPtr(odbn);
  const std::string& idbn = op()->SoleIdbn();
  Blob* out_blob = BnInOp2BlobPtr(idbn);
  size_t type_size = sizeof(floating_point_type);

  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == copy_hd_conf.H2D) {
    CopyH2DAsync(in_blob, out_blob, ctx, type_size);
  } else {
    CopyD2HAsync(in_blob, out_blob, ctx, type_size);
  }
}

INSTANTIATE_GPU_KERNEL_CLASS(CopyHdKernel);
REGISTER_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);

}  // namespace oneflow
