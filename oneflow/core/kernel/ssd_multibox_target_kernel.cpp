#include "oneflow/core/kernel/ssd_multibox_target_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"

namespace oneflow {

template<typename T>
void SSDMultiboxTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearOutputBlobs(ctx, BnInOp2Blob);
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  SingleThreadLoop(bbox_blob->shape().At(0), [&](int64_t im_i) {
    auto boxes = CalcBoxesAndGtBoxesMaxOverlaps(im_i, BnInOp2Blob);
    RegionProposal(im_i, BnInOp2Blob);
    ApplyNms(im_i, BnInOp2Blob);
  });
  // FOR_RANGE(int32_t, i, 0, BnInOp2Blob("bbox")->shape().At(0)) {
  //   auto gt_boxes = GetImageGtBoxes(i, BnInOp2Blob);
  //   auto boxes = GetImageRoiBoxes(i, BnInOp2Blob);
  //   ComputeRoiBoxesAndGtBoxesOverlaps(gt_boxes, boxes, BnInOp2Blob("overlaps")->mut_dptr<float>());
  //   int32_t pos_num = SelectPosNegSample(ctx, i, boxes, gt_boxes, BnInOp2Blob);
  //   Output(i, boxes, gt_boxes, pos_num, BnInOp2Blob);
  // }
}

template<typename T>
void SSDMultiboxTargetKernel<T>::ClearOutputBlobs(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* sampled_indices_blob = BnInOp2Blob("sampled_indices");
  Blob* positive_sampled_indices_blob = BnInOp2Blob("positive_sampled_indices");
  Blob* bbox_labels_blob = BnInOp2Blob("bbox_labels");
  Blob* bbox_deltas_blob = BnInOp2Blob("bbox_deltas");
  Blob* bbox_inside_weights_blob = BnInOp2Blob("bbox_inside_weights");
  Blob* bbox_outside_weights_blob = BnInOp2Blob("bbox_outside_weights");

  std::memset(sampled_indices_blob->mut_dptr(), 0, 
              sampled_indices_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(positive_sampled_indices_blob->mut_dptr(), 0, 
              positive_sampled_indices_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(bbox_labels_blob->mut_dptr(), 0, 
              bbox_labels_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(bbox_deltas_blob->mut_dptr(), 0, 
              bbox_deltas_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_inside_weights_blob->mut_dptr(), 0,
              bbox_inside_weights_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_outside_weights_blob->mut_dptr(), 0,
              bbox_outside_weights_blob->shape().elem_cnt() * sizeof(T));
}

template<typename T>
typename SSDMultiboxTargetKernel<T>::BoxesWithMaxOverlapSlice
void SSDMultiboxTargetKernel<T>::CalcBoxesAndGtBoxesMaxOverlaps(
    int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const SSDMultiBoxTargetOpConf& conf = op_conf().ssd_multibox_target_conf();
  // Col gt boxes
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const size_t col_num = gt_boxes_blob->dim1_valid_num(im_index);
  const BBox* gt_boxes = BBox::Cast(gt_boxes_blob->dptr<T>(im_index));
  // Row boxes 
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  Blob* sampled_indices_blob = BnInOp2Blob("sampled_indices");
  const size_t row_num = bbox_blob->has_dim1_valid_num_field() ? 
      bbox_blob->dim1_valid_num(im_index) : bbox_blob->shape().At(1);
  BoxesWithMaxOverlapSlice boxes(
      BoxesSlice(IndexSequence(sampled_indices_blob->shape().At(1), row_num,
                               sampled_indices_blob->mut_dptr<int32_t>(im_index), true),
                 bbox_blob->dptr<T>()),
      BnInOp2Blob("max_overlaps")->mut_dptr<float>(im_index),
      BnInOp2Blob("max_overlaps_gt_index")->mut_dptr<int32_t>(im_index), true),

  float max_overlap = 0.0f;
  int32_t max_overlap_index = -1;
  std::list<int32_t> overlap_index_list;
  std::vector<float> overlap_vec(row_num * col_num);

  FOR_RANGE(size_t, i, 0, row_num) {
    FOR_RANGE(size_t, j, 0, col_num) {
      const BBox* gt_bbox = gt_boxes + j;
      if (gt_bbox->Area() <= 0) { continue; }
      const float overlap = boxes.GetBBox(i)->InterOverUnion(gt_bbox);
      const int32_t index = i * col_num + j;
      overlap_vec[index] = overlap;
      overlap_index_list.emplace_back(index);
      if (overlap > max_overlap) {
        max_overlap = overlap;
        max_overlap_index = index;
      }
      int32_t max_overlap_gt_index = overlap >= conf.positive_overlap_threshold() ? j : -1;
      boxes.TryUpdateMaxOverlap(boxes.GetIndex(i), max_overlap_gt_index, overlap);
    }
  }
  
  size_t loop_cnt = col_num;
  while (loop_cnt--) {
    float cur_max_overlap = 0.0f;
    int32_t cur_max_overlap_index = -1;
    int32_t i = max_overlap_index / col_num;
    int32_t j = max_overlap_index % col_num;
    if (max_overlap_index >= 0) {
      boxes.set_max_overlap(i, max_overlap);
      boxes.set_max_overlap_with_index(i, j);
    }
    for (auto itor = overlap_index_list.begin(); itor != overlap_index_list.end();) {
      int32_t overlap_index = *itor;
      if ((max_overlap_index >= 0) && 
          (overlap_index % col_num == j || overlap_index / col_num == i)) {
        itor = overlap_index_list.erase(itor);
      } else {
        if (overlap_vec[overlap_index] > cur_max_overlap) {
          cur_max_overlap = overlap_vec[overlap_index];
          cur_max_overlap_index = overlap_index; 
        }
        ++itor;
      }
    }
    max_overlap = cur_max_overlap;
    max_overlap_index = cur_max_overlap_index;
  }

  return boxes
}

template<typename T>
void SSDMultiboxTargetKernel<T>::SamplePositiveAndNegtive(
    const int64_t im_index, const BoxesWithMaxOverlapSlice& boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* positive_sampled_indices_blob = BnInOp2Blob("positive_sampled_indices");
  Blob* negtive_sampled_indices_blob = BnInOp2Blob("negtive_sampled_indices");
  IndexSequence pos_sample_inds(positive_sampled_indices_blob->shape().At(1),
                                positive_sampled_indices_blob->mut_dptr<int32_t>(im_index), false);
  IndexSequence neg_sample_inds(negtive_sampled_indices_blob->shape().At(1),
                                negtive_sampled_indices_blob->mut_dptr<int32_t>(im_index), false);
  pos_sample_inds.Assign(boxes);
  pos_sample_inds.Filter([&](int32_t index) { return boxes.max_overlap_with_index(index) < 0; });

  // CalcSoftmaxLoss(ctx, im_index, roi_boxes, gt_boxes, BnInOp2Blob);
  // neg_sample.Assign(roi_boxes);
  // neg_sample.Filter([&](size_t n, int32_t index) {
  //   return roi_boxes.max_overlap_gt_index(index) != -1
  //          || roi_boxes.max_overlap(index) >= conf.neg_overlap();
  // });
  // auto neg_sample_with_score = ScoresIndex<T>(neg_sample, BnInOp2Blob("loss_score")->mut_dptr<T>());
  // neg_sample_with_score.Sort([&](int32_t lhs_index, int32_t rhs_index) {
  //   return neg_sample_with_score.score(lhs_index) > neg_sample_with_score.score(rhs_index);
  // });
  // int32_t num_neg = std::min(static_cast<int32_t>(neg_sample_with_score.size()),
  //                            static_cast<int32_t>(num_pos * conf.neg_pos_ratio()));
  // neg_sample_with_score.Truncate(num_neg);
  // neg_sample.Assign(neg_sample_with_score);
}

template<typename T>
void SSDMultiboxTargetKernel<T>::CalcSoftmaxLoss(
    const KernelCtx& ctx, size_t im_index, const BoxesWithMaxOverlap& roi_boxes,
    const GtBoxesAndLabels& gt_boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  // score_blob(r, class_num)->prob(r, 81)->loss_score_blob(r)
  // const Blob* score_blob = BnInOp2Blob("score");                        //(n,r,c)
  // Blob* buf_blob = BnInOp2Blob("fw_buf");                               //(r,c)
  // const T* pred = score_blob->dptr<T>(im_index);                        //(n,r,c)
  // int32_t* label = BnInOp2Blob("labels")->mut_dptr<int32_t>(im_index);  //(r)
  // T* prob = BnInOp2Blob("prob")->mut_dptr<T>();                         //(r,c)
  // T* loss = BnInOp2Blob("loss_score")->mut_dptr<T>();                   //(r)
  // FOR_RANGE(int32_t, index, 0, score_blob->shape().At(1)) {
  //   int32_t gt_box_index = roi_boxes.max_overlap_gt_index(index);
  //   label[index] = gt_box_index >= 0 ? gt_boxes.GetLabel(gt_box_index) : 0;
  // }
  // const int64_t n = score_blob->shape().At(1);
  // const int64_t w = score_blob->shape().At(2);
  // SoftmaxComputeProb<DeviceType::kCPU, T>(ctx.device_ctx, n, w, pred, loss, prob,
  //                                         buf_blob->mut_dptr(),
  //                                         buf_blob->ByteSizeOfDataContentField());
  // SparseCrossEntropyLossKernelUtil<DeviceType::kCPU, T, int32_t>::Forward(ctx.device_ctx, n, w,
  //                                                                         prob, label, loss);
}

// template<typename T>
// int32_t MultiboxOutKernel<T>::SelectPosNegSample(
//     const KernelCtx& ctx, size_t im_index, BoxesWithMaxOverlap& roi_boxes,
//     const GtBoxesAndLabels& gt_boxes,
//     const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
//   Indexes pos_sample(BnInOp2Blob("bbox")->shape().At(1),
//                      BnInOp2Blob("pos_index")->mut_dptr<int32_t>());
//   SelectPositive(roi_boxes, pos_sample);
//   Indexes neg_sample(BnInOp2Blob("bbox")->shape().At(1),
//                      BnInOp2Blob("neg_index")->mut_dptr<int32_t>());
//   SelectNegtive(ctx, im_index, pos_sample.size(), roi_boxes, gt_boxes, neg_sample, BnInOp2Blob);
//   roi_boxes.Truncate(0);
//   roi_boxes.Concat(pos_sample);
//   roi_boxes.Concat(neg_sample);
//   return pos_sample.size();
// }

// template<typename T>
// void MultiboxOutKernel<T>::SelectPositive(const BoxesWithMaxOverlap& roi_boxes,
//                                           Indexes& pos_sample) const {
//   pos_sample.Assign(roi_boxes);
//   pos_sample.Filter(
//       [&](size_t n, int32_t index) { return roi_boxes.max_overlap_gt_index(index) < 0; });
// }

// template<typename T>
// void MultiboxOutKernel<T>::SelectNegtive(
//     const KernelCtx& ctx, size_t im_index, const int32_t num_pos,
//     const BoxesWithMaxOverlap& roi_boxes, const GtBoxesAndLabels& gt_boxes, Indexes& neg_sample,
//     const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
//   const MultiboxOutOpConf& conf = op_conf().multibox_out_conf();
//   ComputeSoftmaxLoss(ctx, im_index, roi_boxes, gt_boxes, BnInOp2Blob);
//   neg_sample.Assign(roi_boxes);
//   neg_sample.Filter([&](size_t n, int32_t index) {
//     return roi_boxes.max_overlap_gt_index(index) != -1
//            || roi_boxes.max_overlap(index) >= conf.neg_overlap();
//   });
//   auto neg_sample_with_score = ScoresIndex<T>(neg_sample, BnInOp2Blob("loss_score")->mut_dptr<T>());
//   neg_sample_with_score.Sort([&](int32_t lhs_index, int32_t rhs_index) {
//     return neg_sample_with_score.score(lhs_index) > neg_sample_with_score.score(rhs_index);
//   });
//   int32_t num_neg = std::min(static_cast<int32_t>(neg_sample_with_score.size()),
//                              static_cast<int32_t>(num_pos * conf.neg_pos_ratio()));
//   neg_sample_with_score.Truncate(num_neg);
//   neg_sample.Assign(neg_sample_with_score);
// }

// template<typename T>
// void MultiboxOutKernel<T>::ComputeSoftmaxLoss(
//     const KernelCtx& ctx, size_t im_index, const BoxesWithMaxOverlap& roi_boxes,
//     const GtBoxesAndLabels& gt_boxes,
//     const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
//   // score_blob(r, class_num)->prob(r, 81)->loss_score_blob(r)
//   const Blob* score_blob = BnInOp2Blob("score");                        //(n,r,c)
//   Blob* buf_blob = BnInOp2Blob("fw_buf");                               //(r,c)
//   const T* pred = score_blob->dptr<T>(im_index);                        //(n,r,c)
//   int32_t* label = BnInOp2Blob("labels")->mut_dptr<int32_t>(im_index);  //(r)
//   T* prob = BnInOp2Blob("prob")->mut_dptr<T>();                         //(r,c)
//   T* loss = BnInOp2Blob("loss_score")->mut_dptr<T>();                   //(r)
//   FOR_RANGE(int32_t, index, 0, score_blob->shape().At(1)) {
//     int32_t gt_box_index = roi_boxes.max_overlap_gt_index(index);
//     label[index] = gt_box_index >= 0 ? gt_boxes.GetLabel(gt_box_index) : 0;
//   }
//   const int64_t n = score_blob->shape().At(1);
//   const int64_t w = score_blob->shape().At(2);
//   SoftmaxComputeProb<DeviceType::kCPU, T>(ctx.device_ctx, n, w, pred, loss, prob,
//                                           buf_blob->mut_dptr(),
//                                           buf_blob->ByteSizeOfDataContentField());
//   SparseCrossEntropyLossKernelUtil<DeviceType::kCPU, T, int32_t>::Forward(ctx.device_ctx, n, w,
//                                                                           prob, label, loss);
// }

// template<typename T>
// void MultiboxOutKernel<T>::Output(int32_t im_index, const BoxesWithMaxOverlap& roi_boxes,
//                                   const GtBoxesAndLabels& gt_boxes, const int32_t pos_num,
//                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
//   const MultiboxOutOpConf& conf = op_conf().multibox_out_conf();
//   int32_t class_num = conf.num_classes();
//   const BBoxDelta<T>* bbox_delta = BBoxDelta<T>::Cast(BnInOp2Blob("bbox_delta")->dptr<T>(im_index));
//   const Blob* score_blob = BnInOp2Blob("score");  //(n,r,c)
//   const T* score_ptr = score_blob->dptr<T>(im_index);
//   T* out_scores_ptr = BnInOp2Blob("out_scores")->mut_dptr<T>(im_index);
//   int32_t* labels_ptr = BnInOp2Blob("labels")->mut_dptr<int32_t>(im_index);
//   int32_t* valid_num_ptr = BnInOp2Blob("valid_num")->mut_dptr<int32_t>();
//   int32_t* pos_num_ptr = BnInOp2Blob("pos_num")->mut_dptr<int32_t>();
//   int32_t* sample_num_ptr = BnInOp2Blob("sample_num")->mut_dptr<int32_t>();
//   BBoxDelta<T>* bbox_preds =
//       BBoxDelta<T>::MutCast(BnInOp2Blob("bbox_preds")->mut_dptr<T>(im_index));
//   BBoxDelta<T>* bbox_targets =
//       BBoxDelta<T>::MutCast(BnInOp2Blob("bbox_targets")->mut_dptr<T>(im_index));
//   T* bbox_inside_weights_ptr = BnInOp2Blob("bbox_inside_weights")->mut_dptr<T>(im_index);
//   T* bbox_outside_weights_ptr = BnInOp2Blob("bbox_outside_weights")->mut_dptr<T>(im_index);
//   valid_num_ptr[im_index] = roi_boxes.size();
//   sample_num_ptr[im_index] = valid_num_ptr[im_index];
//   pos_num_ptr[im_index] = pos_num;
//   FOR_RANGE(int32_t, i, 0, roi_boxes.size()) {
//     int32_t index = roi_boxes.GetIndex(i);
//     int32_t gt_index = roi_boxes.GetMaxOverlapGtIndex(i);
//     if (gt_index >= 0) {
//       const auto* rois_bbox = roi_boxes.GetBBox(i);
//       CopyBboxDelta(&bbox_delta[index], &bbox_preds[i]);
//       bbox_targets[i].TransformInverse(rois_bbox, gt_boxes.GetBBox<float>(gt_index),
//                                        conf.bbox_reg_weights(), false);
//       labels_ptr[i] = gt_boxes.GetLabel(gt_index);
//       bbox_inside_weights_ptr[i * 4] = 1.0f;
//       bbox_inside_weights_ptr[i * 4 + 1] = 1.0f;
//       bbox_inside_weights_ptr[i * 4 + 2] = 1.0f;
//       bbox_inside_weights_ptr[i * 4 + 3] = 1.0f;
//       bbox_outside_weights_ptr[i * 4] = 1.0f / pos_num;
//       bbox_outside_weights_ptr[i * 4 + 1] = 1.0f / pos_num;
//       bbox_outside_weights_ptr[i * 4 + 2] = 1.0f / pos_num;
//       bbox_outside_weights_ptr[i * 4 + 3] = 1.0f / pos_num;
//     } else {
//       labels_ptr[i] = 0;
//     }
//     CopyElements(class_num, score_ptr + index * class_num, out_scores_ptr + i * class_num);
//   }
//   LOG(INFO) << "for gdb";
// }

// template<typename T>
// void MultiboxOutKernel<T>::CopyBboxDelta(const BBoxDelta<T>* input, BBoxDelta<T>* output) const {
//   output->set_dx(input->dx());
//   output->set_dy(input->dy());
//   output->set_dw(input->dw());
//   output->set_dh(input->dh());
// }

// template<typename T>
// void MultiboxOutKernel<T>::CopyElements(const int32_t num, const T* input, T* output) const {
//   FOR_RANGE(int32_t, i, 0, num) { output[i] = input[i]; }
// }

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kSsdMultiboxTargetConf, SSDMultiboxTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
