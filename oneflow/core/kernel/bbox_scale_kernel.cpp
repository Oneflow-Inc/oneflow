#include "oneflow/core/kernel/bbox_scale_kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

namespace {

template<typename T>
void ScaleBBox(const BBoxT<T>* origin_bbox, const T x_scale, const T y_scale, BBoxT<T>* ret_bbox) {
  ret_bbox->set_ltrb(origin_bbox->left() * x_scale, origin_bbox->top() * y_scale,
                     origin_bbox->right() * x_scale, origin_bbox->bottom() * y_scale);
}

}  // namespace

template<typename T>
void BboxScaleKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_box = BnInOp2Blob("in");
  const Blob* scale = BnInOp2Blob("scale");
  Blob* out_box = BnInOp2Blob("out");
  Memset<DeviceType::kCPU>(ctx.device_ctx, out_box->mut_dptr<T>(), 0,
                           out_box->ByteSizeOfDataContentField());

  const T* scale_ptr = scale->dptr<T>();
  const auto* in_bbox_ptr = BBoxT<T>::Cast(in_box->dptr<T>());
  auto* out_bbox_ptr = BBoxT<T>::Cast(out_box->mut_dptr<T>());
  const int32_t dims = out_box->static_shape().NumAxes();
  if (dims == 2) {
    FOR_RANGE(int32_t, i, 0, out_box->shape().At(0)) {
      const int32_t im_idx = in_box->record_id_in_device_piece(i);
      const T y_scale = scale_ptr[im_idx * 2 + 0];
      const T x_scale = scale_ptr[im_idx * 2 + 1];
      ScaleBBox(in_bbox_ptr + i, x_scale, y_scale, out_bbox_ptr + i);
    }
  } else if (dims == 3) {
    FOR_RANGE(int32_t, i, 0, out_box->shape().At(0)) {
      const T y_scale = scale_ptr[i * 2 + 0];
      const T x_scale = scale_ptr[i * 2 + 1];
      FOR_RANGE(int32_t, j, 0, out_box->dim1_valid_num(i)) {
        const int32_t bbox_offset = i * out_box->static_shape().At(1) + j;
        ScaleBBox(in_bbox_ptr + bbox_offset, x_scale, y_scale, out_bbox_ptr + bbox_offset);
      }
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void BboxScaleKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->set_dim0_valid_num(0, BnInOp2Blob("in")->dim0_valid_num(0));
}

template<typename T>
void BboxScaleKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim1ValidNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<typename T>
void BboxScaleKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyRecordIdInDevicePieceFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxScaleConf, BboxScaleKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
