#include "oneflow/core/kernel/box_scale_kernel.h"

namespace oneflow {

template<typename T>
void BoxScaleKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_box = BnInOp2Blob("in_box");
  const Blob* scale = BnInOp2Blob("scale");
  Blob* out_box = BnInOp2Blob("out_box");
  const T* in_box_ptr = in_box->dptr<T>();
  T* out_box_ptr = out_box->mut_dptr<T>();
  Memset<DeviceType::kCPU>(ctx.device_ctx, out_box_ptr, 0, out_box->ByteSizeOfDataContentField());

  const bool dim0_varing = in_box->has_dim0_valid_num_field();
  const bool dim1_varing = in_box->has_dim1_valid_num_field();
  if (!dim0_varing && dim1_varing) {
    // in_box: (N, G, 4)
    int32_t max_num_bbox_per_image = in_box->static_shape().At(1);
    const int32_t num_imgs = in_box->shape().At(0);
    CHECK_EQ(num_imgs, scale->shape().At(0));
    FOR_RANGE(int64_t, im_idx, 0, num_imgs) {
      T scale_y = scale->dptr<T>(im_idx)[0];
      T scale_x = scale->dptr<T>(im_idx)[1];
      int32_t num_box_per_img = 0;
      FOR_RANGE(int64_t, j, 0, in_box->dim1_valid_num(im_idx)) {
        const int32_t offset = (im_idx * max_num_bbox_per_image + j) * 4;
        T in_x_min = in_box_ptr[offset + 0];
        T in_y_min = in_box_ptr[offset + 1];
        T in_x_max = in_box_ptr[offset + 2];
        T in_y_max = in_box_ptr[offset + 3];
        T* cur_out_box_ptr = out_box_ptr + (im_idx * max_num_bbox_per_image + num_box_per_img) * 4;
        if (in_x_max > in_x_min && in_y_max > in_y_min) {
          cur_out_box_ptr[0] = in_x_min * scale_x;
          cur_out_box_ptr[1] = in_y_min * scale_y;
          cur_out_box_ptr[2] = in_x_max * scale_x;
          cur_out_box_ptr[3] = in_y_max * scale_y;
          num_box_per_img += 1;
        }
      }
      out_box->set_dim1_valid_num(im_idx, num_box_per_img);
    }
  } else if (dim0_varing && !dim1_varing) {
    // in_box: (R, 4)
    CHECK(in_box->has_record_id_in_device_piece_field());
    CHECK(out_box->has_record_id_in_device_piece_field());
    int32_t num_box = 0;
    FOR_RANGE(int32_t, i, 0, in_box->shape().At(0)) {
      const int32_t im_idx = in_box->record_id_in_device_piece(i);
      const T x_scale = scale->dptr<T>(im_idx)[0];
      const T y_scale = scale->dptr<T>(im_idx)[1];
      const int32_t offset = i * 4;
      T in_x_min = in_box_ptr[offset + 0];
      T in_y_min = in_box_ptr[offset + 1];
      T in_x_max = in_box_ptr[offset + 2];
      T in_y_max = in_box_ptr[offset + 3];
      if (in_x_max > in_x_min && in_y_max > in_y_min) {
        out_box_ptr[i * 4] = in_x_min * x_scale;
        out_box_ptr[i * 4 + 1] = in_y_min * y_scale;
        out_box_ptr[i * 4 + 2] = in_x_min * x_scale;
        out_box_ptr[i * 4 + 3] = in_y_min * y_scale;
        num_box += 1;
      }
      out_box->set_record_id_in_device_piece(i, im_idx);
    }
    out_box->set_dim0_valid_num(0, num_box);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void BoxScaleKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Do nothing
}

template<typename T>
void BoxScaleKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Do nothing
}

template<typename T>
void BoxScaleKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Do nothing
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxScaleConf, BoxScaleKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
