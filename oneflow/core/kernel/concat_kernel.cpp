#include "oneflow/core/kernel/concat_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
FloatingPointType* CalcCopyAddr(FloatingPointType* start_addr, int64_t cp_times,
                                int64_t axis_dim, int64_t offset,
                                int64_t elem_cnt) {
  return start_addr + (cp_times * axis_dim + offset) * elem_cnt;
}

}  // namespace

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::ConcatKernelWork(
    const KernelCtx& ctx, const std::string& out_bn,
    const std::vector<std::string>& in_bns,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    MemCopyFuncType copy_func) const {
  Blob* out_blob = BnInOp2BlobPtr(out_bn);
  if (in_bns.size() == 0) { return; }
  const int32_t concat_axis = op()->op_conf().concat_conf().axis();
  int64_t concat_element_cnt = 1;
  if ((concat_axis != (out_blob->shape().NumAxes() - 1))
      && (concat_axis != -1)) {
    concat_element_cnt = out_blob->shape().Count(concat_axis + 1);
  }
  int64_t concat_num_each_blob = 1;
  if ((concat_axis != (-out_blob->shape().NumAxes())) && (concat_axis != 0)) {
    concat_num_each_blob = out_blob->shape().Count(0, concat_axis);
  }
  const int64_t out_concat_axis_dim = out_blob->shape().At(concat_axis);
  FloatingPointType* out_blob_dptr = out_blob->mut_dptr<FloatingPointType>();
  int64_t offset_concat_axis = 0;
  cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyHostToHost;
  if (device_type == DeviceType::kGPU) {
    kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  }

  for (const std::string& in_bn : in_bns) {
    const Blob* in_blob = BnInOp2BlobPtr(in_bn);
    FloatingPointType* in_blob_dptr = in_blob->mut_dptr<FloatingPointType>();
    const int64_t in_concat_axis_dim = in_blob->shape().At(concat_axis);
    const int64_t cp_sz =
        in_concat_axis_dim * concat_element_cnt * sizeof(FloatingPointType);

    for (int64_t concat_idx = 0; concat_idx < concat_num_each_blob;
         ++concat_idx) {
      FloatingPointType* out_cp_adr =
          CalcCopyAddr(out_blob_dptr, concat_idx, out_concat_axis_dim,
                       offset_concat_axis, concat_element_cnt);
      FloatingPointType* in_cp_adr = CalcCopyAddr(
          in_blob_dptr, concat_idx, in_concat_axis_dim, 0, concat_element_cnt);
      copy_func(ctx, in_cp_adr, out_cp_adr, cp_sz, kind);
    }

    offset_concat_axis += in_concat_axis_dim;
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  auto copy_in2out = [](const KernelCtx& ctx, FloatingPointType* src,
                        FloatingPointType* dst, const int64_t size,
                        cudaMemcpyKind kind) {
    KernelUtil<device_type, FloatingPointType>::Memcpy(ctx, dst, src, size,
                                                       kind);
  };
  ConcatKernelWork(ctx, op()->SoleObn(), op()->input_bns(), BnInOp2BlobPtr,
                   copy_in2out);
}

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  auto copy_out2in = [](const KernelCtx& ctx, FloatingPointType* dst,
                        FloatingPointType* src, const int64_t size,
                        cudaMemcpyKind kind) {
    KernelUtil<device_type, FloatingPointType>::Memcpy(ctx, dst, src, size,
                                                       kind);
  };
  ConcatKernelWork(ctx, op()->SoleOdbn(), op()->input_diff_bns(),
                   BnInOp2BlobPtr, copy_out2in);
}

INSTANTIATE_KERNEL_CLASS(ConcatKernel);
REGISTER_KERNEL(OperatorConf::kConcatConf, ConcatKernel);

}  // namespace oneflow
