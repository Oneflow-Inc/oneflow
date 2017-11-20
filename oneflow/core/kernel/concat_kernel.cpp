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
    const KernelCtx& ctx, const std::string& obn,
    const std::vector<std::string>& ibns,
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    MemCopyFuncType copy_func) const {
  Blob* out_blob = BnInOp2Blob(obn);
  if (ibns.size() == 0) { return; }
  const int32_t concat_axis = this->op_conf().concat_conf().axis();
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

  for (const std::string& ibn : ibns) {
    Blob* in_blob = BnInOp2Blob(ibn);
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
  if (BnInOp2Blob(ibns.front())->has_data_id()) {
    CopyDataIdToOb(ctx, ibns, obn, concat_axis, kind, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::CopyDataIdToOb(
    const KernelCtx& ctx, const std::vector<std::string>& ibns,
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

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_in2out = [](const KernelCtx& ctx, T* src, T* dst,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(ctx, this->kernel_conf().output_bns(0),
                   this->kernel_conf().input_bns(), BnInOp2Blob, copy_in2out);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto copy_out2in = [](const KernelCtx& ctx, T* dst, T* src,
                        const int64_t size, cudaMemcpyKind kind) {
    Memcpy<device_type>(ctx.device_ctx, dst, src, size, kind);
  };
  ConcatKernelWork(ctx, this->kernel_conf().output_diff_bns(0),
                   this->kernel_conf().input_diff_bns(), BnInOp2Blob,
                   copy_out2in);
}

}  // namespace oneflow
