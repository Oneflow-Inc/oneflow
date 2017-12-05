#include "oneflow/core/kernel/concat_kernel.h"

namespace oneflow {

namespace {

// Calculates the address of a given position in a shape's concat dim.
// Consider a blob with shape (2,3,4,5,6) and concat axis is the 3rd dim. Taking
// it as a multi-dim array, when we want to calculate the address of
// blob[1][1][2][0][0]. Parameters are setted as: concat_idx = 5,
// concat_axis_dim = 4, concat_axis_offset = 2 and concat_elem_bytesize =
// 30 * sizeof(data type in blob)
char* NextConcatAddr(char* start_addr, int64_t concat_idx,
                     int64_t concat_axis_dim, int64_t concat_axis_offset,
                     int64_t concat_elem_bytesize) {
  return start_addr
         + (concat_idx * concat_axis_dim + concat_axis_offset)
               * concat_elem_bytesize;
}

template<DeviceType device_type>
struct GetCudaMemcpyKind;

template<>
struct GetCudaMemcpyKind<DeviceType::kCPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyHostToHost;
};

template<>
struct GetCudaMemcpyKind<DeviceType::kGPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};

}  // namespace

template<DeviceType device_type>
void ConcatKernel<device_type>::ConcatKernelWork(
    const KernelCtx& ctx, const std::string& obn,
    const PbRpf<std::string>& ibns,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (ibns.size() == 0) { return; }

  Blob* out_blob = BnInOp2Blob(obn);
  const int32_t concat_axis = this->op_conf().concat_conf().axis();
  const int64_t out_concat_axis_dim = out_blob->shape().At(concat_axis);
  const int64_t total_cp_num = this->kernel_conf().concat_conf().total_cp_num();
  char* out_blob_mut_dptr = out_blob->mut_dptr<char>();

  cudaMemcpyKind kind = GetCudaMemcpyKind<device_type>::val;
  int64_t concat_axis_offset = 0;
  int64_t cp_dim_bytesize = 0;
  const PbRf<int64_t>& per_cp_bytesize =
      this->kernel_conf().concat_conf().per_cp_bytesize();
  FOR_RANGE(int64_t, ibn_idx, 0, ibns.size()) {
    Blob* in_blob = BnInOp2Blob(ibns[ibn_idx]);
    char* in_blob_mut_dptr = in_blob->mut_dptr<char>();
    const int64_t in_concat_axis_dim = in_blob->shape().At(concat_axis);
    const int64_t cp_bytesize = per_cp_bytesize[ibn_idx];
    if (cp_dim_bytesize == 0) {
      cp_dim_bytesize = cp_bytesize / in_concat_axis_dim;
    }
    FOR_RANGE(int64_t, concat_idx, 0, total_cp_num) {
      char* out_cp_adr =
          NextConcatAddr(out_blob_mut_dptr, concat_idx, out_concat_axis_dim,
                         concat_axis_offset, cp_dim_bytesize);
      char* in_cp_adr = NextConcatAddr(in_blob_mut_dptr, concat_idx,
                                       in_concat_axis_dim, 0, cp_dim_bytesize);
      if (this->kernel_conf().is_forward()) {
        Memcpy<device_type>(ctx.device_ctx, out_cp_adr, in_cp_adr, cp_bytesize,
                            kind);
      } else {
        Memcpy<device_type>(ctx.device_ctx, in_cp_adr, out_cp_adr, cp_bytesize,
                            kind);
      }
    }
    concat_axis_offset += in_concat_axis_dim;
  }
}

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ConcatKernelWork(ctx, this->kernel_conf().output_bns(0),
                   this->kernel_conf().input_bns(), BnInOp2Blob);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob(this->kernel_conf().output_bns(0));
  const Blob* in_blob_0 = BnInOp2Blob(this->kernel_conf().input_bns(0));
  out_blob->CopyDataIdFrom<device_type>(ctx.device_ctx, in_blob_0);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ConcatKernelWork(ctx, this->kernel_conf().output_diff_bns(0),
                   this->kernel_conf().input_diff_bns(), BnInOp2Blob);
}

namespace {

Kernel* CreateConcatKernel(DeviceType dev_type) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define CONCAT_KERNEL_ENTRY(device_type) \
  {GetHashKey(device_type), []() { return new ConcatKernel<device_type>; }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(CONCAT_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(GetHashKey(dev_type))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kConcatConf, CreateConcatKernel));

}  // namespace oneflow
