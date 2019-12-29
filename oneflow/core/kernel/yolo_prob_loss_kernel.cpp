#include "oneflow/core/kernel/yolo_prob_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void YoloProbLossKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Memset<device_type>(ctx.device_ctx, BnInOp2Blob("bbox_objness_out")->mut_dptr(), 0,
                      BnInOp2Blob("bbox_objness_out")->shape().elem_cnt() * sizeof(T));
  Memset<device_type>(ctx.device_ctx, BnInOp2Blob("bbox_clsprob_out")->mut_dptr(), 0,
                      BnInOp2Blob("bbox_clsprob_out")->shape().elem_cnt() * sizeof(T));
  const size_t pos_num = 50;
  const size_t neg_num = 20000;
  FOR_RANGE(int32_t, im_index, 0, BnInOp2Blob("bbox_objness")->shape().At(0)) {
    // const size_t pos_num = BnInOp2Blob("pos_inds")->dim1_valid_num(im_index);
    // const size_t neg_num = BnInOp2Blob("neg_inds")->dim1_valid_num(im_index);
    const int32_t* pos_inds_ptr = BnInOp2Blob("pos_inds")->dptr<int32_t>()
                                  + im_index * BnInOp2Blob("pos_inds")->shape().Count(1);
    const int32_t* neg_inds_ptr = BnInOp2Blob("neg_inds")->dptr<int32_t>()
                                  + im_index * BnInOp2Blob("neg_inds")->shape().Count(1);
    const int32_t* pos_cls_label_ptr = BnInOp2Blob("pos_cls_label")->dptr<int32_t>()
                                       + im_index * BnInOp2Blob("pos_cls_label")->shape().Count(1);
    const int32_t* valid_num_ptr = BnInOp2Blob("valid_num")->dptr<int32_t>()
                                   + im_index * BnInOp2Blob("valid_num")->shape().Count(1);

    YoloProbLossKernelUtil<device_type, T>::CalcObjnessDiff(
        ctx.device_ctx, pos_num, neg_num, valid_num_ptr, pos_inds_ptr, neg_inds_ptr,
        BnInOp2Blob("bbox_objness")->dptr<T>()
            + im_index * BnInOp2Blob("bbox_objness")->shape().Count(1),
        BnInOp2Blob("bbox_objness_out")->mut_dptr<T>()
            + im_index * BnInOp2Blob("bbox_objness_out")->shape().Count(1));
    YoloProbLossKernelUtil<device_type, T>::CalcClsProbDiff(
        ctx.device_ctx, pos_num, this->op_conf().yolo_prob_loss_conf().num_classes(), valid_num_ptr,
        pos_inds_ptr, pos_cls_label_ptr,
        BnInOp2Blob("bbox_clsprob")->dptr<T>()
            + im_index * BnInOp2Blob("bbox_clsprob")->shape().Count(1),
        BnInOp2Blob("bbox_clsprob_out")->mut_dptr<T>()
            + im_index * BnInOp2Blob("bbox_clsprob_out")->shape().Count(1));
  }
}

template<typename T>
struct YoloProbLossKernelUtil<DeviceType::kCPU, T> {
  static void CalcObjnessDiff(DeviceCtx* ctx, const size_t pos_num, const size_t neg_num,
                              const int32_t* valid_num_ptr, const int32_t* pos_inds_ptr,
                              const int32_t* neg_inds_ptr, const T* bbox_objness_ptr,
                              T* bbox_objness_out_ptr) {
    FOR_RANGE(size_t, i, 0, pos_num) {
      const int32_t box_index = pos_inds_ptr[i];
      bbox_objness_out_ptr[box_index] = bbox_objness_ptr[box_index] - 1;
    }
    FOR_RANGE(size_t, i, 0, neg_num) {
      const int32_t box_index = neg_inds_ptr[i];
      bbox_objness_out_ptr[box_index] = bbox_objness_ptr[box_index];
    }
  }
  static void CalcClsProbDiff(DeviceCtx* ctx, const size_t pos_num, const int32_t num_clsprobs,

                              const int32_t* valid_num_ptr, const int32_t* pos_inds_ptr,
                              const int32_t* pos_cls_label_ptr, const T* bbox_clsprob_ptr,
                              T* bbox_clsprob_out_ptr) {
    FOR_RANGE(size_t, i, 0, pos_num) {
      const int32_t box_index = pos_inds_ptr[i];
      FOR_RANGE(size_t, j, 0, num_clsprobs) {
        (bbox_clsprob_out_ptr + num_clsprobs * box_index)[j] =
            (bbox_clsprob_ptr + num_clsprobs * box_index)[j];
      }
      if (pos_cls_label_ptr[box_index] >= 0) {
        int32_t idx = num_clsprobs * box_index + pos_cls_label_ptr[box_index];
        bbox_clsprob_out_ptr[idx]--;
      }
    }
  }
};

#define INSTANTIATE_YOLO_PROB_LOSS_KERNEL_UTIL(type_cpp, type_proto) \
  template struct YoloProbLossKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_YOLO_PROB_LOSS_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kYoloProbLossConf, YoloProbLossKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
