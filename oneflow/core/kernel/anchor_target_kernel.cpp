#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

namespace {

template<typename T>
void ForEachOverlapBetweenInsideAnchorsAndGtBoxes(const BBoxSlice<T>& gt_boxes_slice, 
                                            const BBoxSlice<T>& anchor_boxes_slice, 
                                            const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(int32_t, i, 0, gt_boxes_slice.size()) {
    FOR_RANGE(int32_t, j, 0, anchor_boxes_slice.size()) {
      float overlap = anchor_boxes_slice.GetBBox(j)->InterOverUnion(gt_boxes_slice.GetBBox(i));
      Handler(gt_boxes_slice.GetSlice(i), anchor_boxes_slice.GetSlice(j), overlap);
    }
  }
}

void AssignPositiveLabelsToGtBoxesNearestAnchors(const GtBoxesNearestAnchorsInfo& gt_boxes_nearest_anchors,
                                                 AnchorLabelsAndMaxOverlapsInfo& anchor_labels_info) {
  gt_boxes_nearest_anchors.ForEachNearestAnchor([&](int32_t anchor_idx) {
    anchor_labels_info.TrySetPositiveLabel(anchor_idx);
  });
}

}  // namespace

template<typename T>
AnchorLabelsAndMaxOverlapsInfo AnchorTargetKernel<T>::AssignLabels(const BBoxSlice<T>& gt_boxes_slice, 
                                         const BBoxSlice<T>& anchor_boxes_slice, 
                                         const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  // From anchor perspective
  // "anchor_label" (H, W, A)                           label
  // "anchor_max_overlaps" (H, W, A)                    overlap
  // "anchor_max_overlap_gt_boxes_index" (H, W, A)      gt_box_index
  Blob* anchor_labels_blob = BnInOp2Blob("anchor_labels");
  AnchorLabelsAndMaxOverlapsInfo anchor_labels_info(anchor_labels_blob->mut_dptr<T>(),
                                                    BnInOp2Blob("anchor_max_overlaps")->mut_dptr<T>(),
                                                    BnInOp2Blob("anchor_max_overlap_gt_boxes_index")->mut_dptr<T>(),
                                                    GetCustomizedOpConf().positive_overlap_threshold(),
                                                    GetCustomizedOpConf().negative_overlap_threshold(),
                                                    anchor_labels_blob->shape().elem_cnt());
  // From gt_box perspective
  // "gt_boxes_nearest_anchors_index" (max_gt_boxes_num * H * W * A)
  // "gt_max_overlaps" (max_gt_boxes_num, 1)
  GtBoxesNearestAnchorsInfo gt_boxes_nearest_anchors(BnInOp2Blob("gt_boxes_nearest_anchors_index")->mut_dptr<T>(),
                                                     BnInOp2Blob("gt_max_overlaps")->mut_dptr<T>());

  ForEachOverlapBetweenInsideAnchorsAndGtBoxes(gt_boxes_slice, anchor_boxes_slice // For each overlap between anchors and gt_boxes
                                         [&](int32_t gt_box_idx, int32_t anchor_box_idx, float overlap) {
    anchor_labels_info.AssignLabelByOverlapThreshold(anchor_box_idx, gt_box_idx, overlap);
    gt_boxes_nearest_anchors.TryRecordAnchorAsNearest(gt_box_idx, anchor_box_idx, overlap, );
  });
  AssignPositiveLabelsToGtBoxesNearestAnchors(gt_boxes_nearest_anchors, anchor_labels_info);

  return anchor_labels_info;
}

template<typename T>
void LabeledBBoxSlice<T, N>::SubsamplePositiveAndNegativeLabels(LabelBBoxSlice& labeled_anchor_slice) {
  labeled_anchor_slice.GroupByLabel();
  size_t batch_size_per_image = GetCustomizedOpConf().batch_size_per_image();
  float foreground_fraction = GetCustomizedOpConf().foreground_fraction();
  size_t fg_cnt = batch_size_per_image * fg_ratio;
  fg_cnt = labeled_anchor_slice.Subsample(1, fg_cnt);
  size_t bg_cnt = batch_size_per_image - fg_cnt;
  labeled_anchor_slice.Subsample(0, bg_cnt);
}

template<typename T>
const PbMessage& AnchorTargetKernel<T>::GetCustomizedOpConf() const {
  return this->op_conf().anchors_generator_conf();
}

// output blobs:
//   1. "anchors"
//   2. "inside_anchor_index"
//   3. "insie_anchor_num"
// These three output blobs are used for constructing inside_anchors_slice
template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  const AnchorGeneratorConf& anchor_generator_conf = GetCustomizedOpConf().anchors_generator_conf();
  FasterRcnnUtil<T>::GenerateAnchors(anchor_generator_conf, anchors_blob);
  BBoxSlice<T> inside_anchors_slice(anchors_blob->shape().elem_cnt(), anchors_blob->dptr<T>(), BnInOp2Blob("inside_anchor_index")->mut_dptr<T>());
  inside_anchors_slice.Filter([&](const BBox<T>* anchor_box) {
    return anchor_box->x1() < 0 || anchor_box->y1() < 0 || anchor_box->x2() >= anchor_generator_conf.image_width || anchor_box->y2() >= anchor_generator_conf.image_height;
  });
  *(BnInOp2Blob("inside_anchor_num")->mut_dptr<int32_t>()) = inside_anchors_slice.size();
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");    // (N, H, W, A)
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");  // (N, 1)
  Blob* gt_boxes_absolute_blob = BnInOp2Blob("gt_boxes_absolute");  // (max_gt_boxes_num * 4, 1)
  Blob* anchor_boxes_index_blob = BnInOp2Blob("anchor_boxes_index");
  anchor_boxes_index_blob->CopyFrom(BnInOp2Blob("inside_anchor_index"));

  BBoxSlice<T> anchor_boxes_slice(anchors_blob->shape().elem_cnt(), anchors_blob->dptr<T>(), anchor_boxes_index_blob->mut_dptr<T>(), false);
  anchor_boxes_slice.Truncate(*(BnInOp2Blob("inside_anchor_num")->dptr<int32_t>()));

  FOR_RANGE(int64_t, i, 0, images_num) {  //For Each Image
    // Convert ground truth boxes from OFRecord into absolute coordinates (based on 720 * 720 image in current version)
    int32_t boxes_num = FasterRcnnUtil<T>::ConvertGtBoxesToAbsoluteCoord(gt_boxes_blob->dptr<FloatList16>(i), gt_boxes_absolute_blob->mut_dptr<T>());
    // Construct gt_boxes_slice
    BBoxSlice<T> gt_boxes_slice(GetCustomizedOpConf().max_gt_boxes_num(), gt_boxes_absolute_blob->dptr<T>(), BnInOp2Blob("gt_boxes_index")->mut_dptr<T>());
    gt_boxes_slice.Truncate(boxes_num);
    // Assign labels (-1, 0, 1) to all anchors
    AnchorLabelsAndMaxOverlapsInfo anchor_label_and_nearest_gt_box = AssignLabels(gt_boxes_slice, anchor_boxes_slice, BnInOp2Blob); // TODO: refine AnchorLabelsAndMaxOverlapsInfo and GtBoxesNearestAnchorsInfo
    // Subsample
    LabeledBBoxSlice<size_t, 3> labeled_anchor_slice(anchor_boxes_slice, anchor_label_and_nearest_gt_box.GetAnchorLabels());
    SubsamplePositiveAndNegativeLabels(labeled_anchor_slice);
    
    // LabeledBBoxSlice<size_t, 3> labeled_anchor_slice(); // TODO: fix it
    // labeled_anchor_slice.GroupByLabelType();  // TODO: fix it
    // int64_t start = 0;
    // int64_t end = labeled_anchor_slice.;

    // FOR_RANGE(int64_t, i, 0, labeled_anchor_slice.label_type_num()) {
    //   label = labeled_anchor_slice.label_ptr()[i];
    //   int32_t start_index = labeled_anchor_slice.get_label_start_index(label);
    //   int32_t count = labeled_anchor_slice.get_label_cnt(label);
    //   labeled_anchor_slice.shuffle(start_index, start_index + count);
    // }
    // const AnchorTargetOpConf& anchor_target_conf = GetCustomizedOpConf().anchors_target_conf();
    // int32_t train_piece_size = anchor_target_conf.train_piece_size;
    // int32_t fg_fraction = anchor_target_conf.fg_ratio;
    // int32_t default_fg_cnt = train_piece_size * fg_ratio;

    // fg_cnt = labeled_anchor_slice.get_label_cnt(1);
    // bg_cnt = labeled_anchor_slice.get_label_cnt(0);
    // // fg subsample
    // if(fg_cnt > default_fg_cnt ) {
    //   fg_start = labeled_anchor_slice.get_label_start_index(1);
    //   FOR_RANGE(int32_t, i, fg_start, fg_cnt - default_fg_cnt) {
    //     labeled_anchor_slice.label_ptr[i] = -1;
    //   }
    // } else {
    //   int32_t default_bg_cnt = train_piece_size - fg_cnt;
    //   bg_cnt <= default_bg_cnt ? bg_cnt : default_bg_cnt;
    // }
    // // bg subsample
    // bg_start = labeled_anchor_slice.get_label_start_index(0);
    // FOR_RANGE(int32_t, i, bg_start, bg_cnt) {
    //   labeled_anchor_slice.label_ptr[i] = -1;
    // }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
