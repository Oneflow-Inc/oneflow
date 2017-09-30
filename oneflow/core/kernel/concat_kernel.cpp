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
template<typename T>
T* NextConcatAddr(T* start_addr, int64_t concat_idx, int64_t concat_axis_dim,
                  int64_t offset, int64_t elem_cnt) {
  return start_addr + (concat_idx * concat_axis_dim + offset) * elem_cnt;
}

}  // namespace

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ConcatKernelWork(
    const KernelCtx& ctx, const std::string& out_bn,
    const std::vector<std::string>& in_bns,
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    MemCopyFuncType copy_func) const {
  Blob* out_blob = BnInOp2Blob(out_bn);
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
  T* out_blob_mut_dptr = out_blob->mut_dptr<T>();
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

  for (const std::string& in_bn : in_bns) {
    Blob* in_blob = BnInOp2Blob(in_bn);
    T* in_blob_mut_dptr = in_blob->mut_dptr<T>();
    const int64_t in_concat_axis_dim = in_blob->shape().At(concat_axis);
    const int64_t cp_sz = in_concat_axis_dim * concat_element_cnt * sizeof(T);

    for (int64_t concat_idx = 0; concat_idx < concat_num_each_blob;
         ++concat_idx) {
      T* out_cp_adr =
          NextConcatAddr(out_blob_mut_dptr, concat_idx, out_concat_axis_dim,
                         offset_concat_axis, concat_element_cnt);
      T* in_cp_adr = NextConcatAddr(in_blob_mut_dptr, concat_idx,
                                    in_concat_axis_dim, 0, concat_element_cnt);
      copy_func(ctx, in_cp_adr, out_cp_adr, cp_sz, kind);
    }

    offset_concat_axis += in_concat_axis_dim;
  }
  if (BnInOp2Blob(in_bns.front())->has_data_id()) {
    CopyDataIdToOb(ctx, in_bns, out_bn, concat_axis, kind, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::CopyDataIdToOb(
    const KernelCtx& ctx, const std::vector<std::string>& in_bns,
    const std::string& out_bn, const int32_t concat_axis, cudaMemcpyKind kind,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (concat_axis != 0) {
    CopyDataIdToAllOb<device_type>(ctx.device_ctx, BnInOp2Blob,
                                   BnInOp2Blob(in_bns.front()));
    return;
  }
  Blob* out_blob = BnInOp2Blob(out_bn);
  int64_t data_id_offset = 0;
  for (const std::string& in_bn : in_bns) {
    Blob* in_blob = BnInOp2Blob(in_bn);
    CHECK_LE(data_id_offset + in_blob->ByteSizeOfDataIdField(),
             out_blob->TotalByteSize());
    Memcpy<device_type>(
        ctx.device_ctx, out_blob->mut_data_id() + data_id_offset,
        in_blob->data_id(), in_blob->ByteSizeOfDataIdField(), kind);
    data_id_offset += in_blob->ByteSizeOfDataIdField();
  }
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_in2out = [](const KernelCtx& ctx, T* src, T* dst,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(ctx, op()->SoleObn(), op()->input_bns(), BnInOp2Blob,
                   copy_in2out);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_out2in = [](const KernelCtx& ctx, T* dst, T* src,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(ctx, op()->SoleOdbn(), op()->input_diff_bns(), BnInOp2Blob,
                   copy_out2in);
}

Kernel* CreateConcatKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define CONCAT_KERNEL_ENTRY(device_type, data_type_pair)                     \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {        \
     return new ConcatKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>; \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(CONCAT_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       ALL_DATA_TYPE_SEQ)};

  return creators.at(GetHashKey(op_ctx.device_type(),
                                op_ctx.bn_in_op2data_type().at("in_0")))();
}

COMMAND(AddKernelCreator(OperatorConf::kConcatConf, CreateConcatKernel));

}  // namespace oneflow
