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
                     int64_t concat_axis_dim, int64_t offset, int64_t elem_cnt,
                     const size_t elem_size) {
  return start_addr
         + (concat_idx * concat_axis_dim + offset) * elem_cnt * elem_size;
}

}  // namespace

template<DeviceType device_type>
void ConcatKernel<device_type>::ConcatKernelWork(
    bool is_forward, const KernelCtx& ctx, const std::string& obn,
    const PbRpf<std::string>& ibns,
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    MemCopyFuncType copy_func) const {
  const ConcatKernelConf& concat_kernel_conf =
      this->kernel_conf().concat_conf();
  const size_t elem_size = GetSizeOfDataType(concat_kernel_conf.data_type());
  Blob* out_blob = BnInOp2Blob(obn);
  if (ibns.size() == 0) { return; }
  const int32_t concat_axis = this->op_conf().concat_conf().axis();
  const int64_t& concat_element_cnt =
      is_forward ? concat_kernel_conf.fw_concat_element_cnt()
                 : concat_kernel_conf.bw_concat_element_cnt();
  const int64_t& concat_num_each_blob =
      is_forward ? concat_kernel_conf.fw_concat_num_each_blob()
                 : concat_kernel_conf.bw_concat_num_each_blob();
  const int64_t out_concat_axis_dim = out_blob->shape().At(concat_axis);
  char* out_blob_mut_dptr = static_cast<char*>(out_blob->mut_dptr<void>());
  int64_t offset_concat_axis = 0;
  cudaMemcpyKind kind;
  if (device_type == DeviceType::kCPU) {
    kind = cudaMemcpyKind::cudaMemcpyHostToHost;
  } else if (device_type == DeviceType::kGPU) {
    kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  } else {
    LOG(FATAL) << "device type has not been set";
    return;
  }

  size_t index = 0;
  for (const std::string& ibn : ibns) {
    Blob* in_blob = BnInOp2Blob(ibn);
    char* in_blob_mut_dptr = static_cast<char*>(in_blob->mut_dptr<void>());
    const int64_t in_concat_axis_dim = in_blob->shape().At(concat_axis);
    const int64_t cp_sz = is_forward ? concat_kernel_conf.fw_cp_szs()[index]
                                     : concat_kernel_conf.bw_cp_szs()[index];

    for (int64_t concat_idx = 0; concat_idx < concat_num_each_blob;
         ++concat_idx) {
      char* out_cp_adr =
          NextConcatAddr(out_blob_mut_dptr, concat_idx, out_concat_axis_dim,
                         offset_concat_axis, concat_element_cnt, elem_size);
      char* in_cp_adr =
          NextConcatAddr(in_blob_mut_dptr, concat_idx, in_concat_axis_dim, 0,
                         concat_element_cnt, elem_size);
      copy_func(ctx, in_cp_adr, out_cp_adr, cp_sz, kind);
    }

    offset_concat_axis += in_concat_axis_dim;
    index++;
  }
  if (BnInOp2Blob(ibns[0])->has_data_id()) {
    CopyDataIdToOb(ctx, ibns, obn, concat_axis, kind, BnInOp2Blob);
  }
}

template<DeviceType device_type>
void ConcatKernel<device_type>::CopyDataIdToOb(
    const KernelCtx& ctx, const PbRpf<std::string>& ibns,
    const std::string& obn, const int32_t concat_axis, cudaMemcpyKind kind,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
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
void ConcatKernel<device_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_in2out = [](const KernelCtx& ctx, char* src, char* dst,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(true, ctx, this->kernel_conf().output_bns(0),
                   this->kernel_conf().input_bns(), BnInOp2Blob, copy_in2out);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_out2in = [](const KernelCtx& ctx, char* dst, char* src,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(false, ctx, this->kernel_conf().output_diff_bns(0),
                   this->kernel_conf().input_diff_bns(), BnInOp2Blob,
                   copy_out2in);
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
