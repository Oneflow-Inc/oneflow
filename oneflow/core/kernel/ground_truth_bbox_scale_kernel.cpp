#include "oneflow/core/kernel/ground_truth_bbox_scale_kernel.h"

namespace oneflow {

template<typename T>
void GtBboxScaleKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_gt_bbox = BnInOp2Blob("in");
  const Blob* scale = BnInOp2Blob("scale");
  Blob* out_gt_bbox = BnInOp2Blob("out");
  const T* in_gt_bbox_ptr = in_gt_bbox->dptr<T>();
  T* out_gt_bbox_ptr = out_gt_bbox->mut_dptr<T>();
  Memset<DeviceType::kCPU>(ctx.device_ctx, out_gt_bbox_ptr, 0,
                           out_gt_bbox->ByteSizeOfDataContentField());

  int32_t max_num_bbox_per_image = in_gt_bbox->static_shape().At(1);
  FOR_RANGE(int64_t, i, 0, in_gt_bbox->static_shape().At(0)) {
    T scale_y = scale->dptr<T>(i)[0];
    T scale_x = scale->dptr<T>(i)[1];
    int32_t valid_cnt_per_image = 0;
    FOR_RANGE(int64_t, j, 0, in_gt_bbox->dim1_valid_num(i)) {
      int32_t offset = (i * max_num_bbox_per_image + j) * 4;
      T in_x_min = in_gt_bbox_ptr[offset + 0];
      T in_y_min = in_gt_bbox_ptr[offset + 1];
      T in_x_max = in_gt_bbox_ptr[offset + 2];
      T in_y_max = in_gt_bbox_ptr[offset + 3];
      T* cur_out_bbox_ptr =
          out_gt_bbox_ptr + (i * max_num_bbox_per_image + valid_cnt_per_image) * 4;
      if (in_x_max > in_x_min && in_y_max > in_y_min) {
        cur_out_bbox_ptr[0] = in_x_min * scale_x;
        cur_out_bbox_ptr[1] = in_y_min * scale_y;
        cur_out_bbox_ptr[2] = in_x_max * scale_x;
        cur_out_bbox_ptr[3] = in_y_max * scale_y;
        valid_cnt_per_image += 1;
      }
    }
    out_gt_bbox->set_dim1_valid_num(i, valid_cnt_per_image);
  }
}

template<typename T>
void GtBboxScaleKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Do nothing
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kGtBboxScaleConf, GtBboxScaleKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
