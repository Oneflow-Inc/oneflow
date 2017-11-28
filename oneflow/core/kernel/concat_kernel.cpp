#include "oneflow/core/kernel/concat_kernel.h"

namespace oneflow {

namespace {

// Calculates the addr of the next concat point in blob. Ex:
// For a 2-dimension matrix with columns and rows, assume the concat operation
// is on the row dimension, so each row will be concated/extended. The
// start_addr is the address of element(0,0), concat_idx indicates the index
// of / the current row to be concated. concat_axis_dim is the row
// length/dimension. / offset is the concat position in each row. The elem_cnt
// is 1 for 2-dimension / matrix and is calculated by Count(concat_axis,
// NumAxes()) for higher / dimension matrix.
char* NextConcatAddr(char* start_addr, int64_t concat_idx,
                     int64_t concat_axis_dim, int64_t offset,
                     int64_t cp_dim_bytesize) {
  return start_addr + (concat_idx * concat_axis_dim + offset) * cp_dim_bytesize;
}

cudaMemcpyKind GetcudaMemcpyKind(DeviceType device_type) {
  if (device_type == DeviceType::kCPU) {
    return cudaMemcpyKind::cudaMemcpyHostToHost;
  } else {
    CHECK(device_type == DeviceType::kGPU);
    return cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  }
}

}  // namespace

template<DeviceType device_type>
void ConcatKernel<device_type>::ConcatKernelWork(
    const KernelCtx& ctx, const std::string& obn,
    const PbRpf<std::string>& ibns,
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    MemCopyFuncType copy_func) const {
  if (ibns.size() == 0) { return; }

  Blob* out_blob = BnInOp2Blob(obn);
  const int32_t concat_axis = this->op_conf().concat_conf().axis();
  const int64_t out_concat_axis_dim = out_blob->shape().At(concat_axis);
  const int64_t total_cp_num = this->kernel_conf().concat_conf().total_cp_num();
  char* out_blob_mut_dptr = out_blob->mut_dptr<char>();

  cudaMemcpyKind kind = GetcudaMemcpyKind(device_type);
  int64_t offset_concat_axis = 0;
  int64_t cp_dim_bytesize = 0;
  const auto& per_cp_bytesize =
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
                         offset_concat_axis, cp_dim_bytesize);
      char* in_cp_adr = NextConcatAddr(in_blob_mut_dptr, concat_idx,
                                       in_concat_axis_dim, 0, cp_dim_bytesize);
      copy_func(ctx, in_cp_adr, out_cp_adr, cp_bytesize, kind);
    }
    offset_concat_axis += in_concat_axis_dim;
  }
}

template<DeviceType device_type>
void ConcatKernel<device_type>::CopyDataIdToOb(
    const KernelCtx& ctx, const PbRpf<std::string>& ibns,
    const std::string& obn,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  cudaMemcpyKind kind = GetcudaMemcpyKind(device_type);
  Blob* out_blob = BnInOp2Blob(obn);
  int64_t data_id_offset = 0;
  for (const std::string& ibn : ibns) {
    Blob* in_blob = BnInOp2Blob(ibn);
    CHECK_LE(data_id_offset + in_blob->ByteSizeOfDataIdField(),
             out_blob->ByteSizeOfDataIdField());
    Memcpy<device_type>(
        ctx.device_ctx, out_blob->mut_data_id() + data_id_offset,
        in_blob->data_id(), in_blob->ByteSizeOfDataIdField(), kind);
    data_id_offset += in_blob->ByteSizeOfDataIdField();
  }
}

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_in2out = [](const KernelCtx& ctx, char* src, char* dst,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(ctx, this->kernel_conf().output_bns(0),
                   this->kernel_conf().input_bns(), BnInOp2Blob, copy_in2out);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_out2in = [](const KernelCtx& ctx, char* dst, char* src,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(ctx, this->kernel_conf().output_diff_bns(0),
                   this->kernel_conf().input_diff_bns(), BnInOp2Blob,
                   copy_out2in);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->kernel_conf().need_do_data_id()) {
    CopyDataIdToOb(ctx, this->kernel_conf().input_bns(),
                   this->kernel_conf().output_bns(0), BnInOp2Blob);
  }
}

template<DeviceType device_type>
void ConcatKernel<device_type>::BackwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->kernel_conf().need_do_data_id()) {
    CopyDataIdToOb(ctx, this->kernel_conf().input_diff_bns(),
                   this->kernel_conf().output_diff_bns(0), BnInOp2Blob);
  }
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
