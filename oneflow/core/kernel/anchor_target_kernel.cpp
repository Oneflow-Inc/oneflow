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
  // "anchor_label" (H, W, A)                     label
  // "anchor_max_overlaps" (H, W, A)              overlap
  // "anchor_max_overlap_gt_boxes" (H, W, A)      gt_box_index
  AnchorLabelsAndMaxOverlapsInfo anchor_labels_info(BnInOp2Blob("anchor_labels")->mut_dptr<T>(),
                                                    BnInOp2Blob("anchor_max_overlaps")->mut_dptr<T>(),
                                                    BnInOp2Blob("anchor_max_overlap_gt_boxes_index")->mut_dptr<T>(),
                                                    GetCustomizedOpConf().positive_overlap_threshold(),
                                                    GetCustomizedOpConf().negative_overlap_threshold());
  // From gt_box perspective
  // "gt_boxes_nearst_anchors" (max_gt_boxes_num, H * W * A)
  // "gt_boxes_nearst_anchors_cnt" (max_gt_boxes_num, 1)
  GtBoxesNearestAnchorsInfo gt_boxes_nearest_anchors(BnInOp2Blob("gt_boxes_nearest_anchors")->mut_dptr<T>(),, 
                                                     BnInOp2Blob("gt_max_overlaps")->mut_dptr<T>());

  ForEachOverlapBetweenInsideAnchorsAndGtBoxes(gt_boxes_slice, anchor_boxes_slice // For each overlap between anchors and gt_boxes
                                         [&](int32_t gt_box_idx, int32_t anchor_box_idx, float overlap) {
    anchor_labels_info.AssignLabelByOverlapThreshold(anchor_box_idx, gt_box_idx, overlap);
    gt_boxes_nearest_anchors.TryRecordAnchorAsNearest(gt_box_idx, anchor_box_idx, overlap);
  });
  AssignPositiveLabelsToGtBoxesNearestAnchors(gt_boxes_nearest_anchors, anchor_labels_info);

  return anchor_labels_info;
}

template<typename T>
AnchorLabelsAndMaxOverlapsInfo AnchorTargetKernel<T>::RandomSubsample(const BBoxSlice<T>& gt_boxes_slice, 
                                         const BBoxSlice<T>& anchor_boxes_slice, 
                                         const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  anchor_boxes_slice.Sort([](const BBox<T>) {

  });
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
  
  // Construct inside_anchors_slice
  BBoxSlice<T> inside_anchors_slice(anchors_blob->shape().elem_cnt(), anchors_blob->dptr<T>(), BnInOp2Blob("inside_anchor_index")->mut_dptr<T>(), false);
  inside_anchors_slice.Truncate(*(BnInOp2Blob("inside_anchor_num")->dptr<int32_t>()));

  FOR_RANGE(int64_t, i, 0, images_num) {  //For Each Image
    // Convert ground truth boxes from OFRecord into absolute coordinates (based on 720 * 720 image in current version)
    int32_t boxes_num = FasterRcnnUtil<T>::ConvertGtBoxesToAbsoluteCoord(gt_boxes_blob->dptr<FloatList16>(i), gt_boxes_absolute_blob->mut_dptr<T>());
    // Construct gt_boxes_slice
    BBoxSlice<T> gt_boxes_slice(GetCustomizedOpConf().max_gt_boxes_num(), gt_boxes_absolute_blob->dptr<T>(), BnInOp2Blob("gt_boxes_index")->mut_dptr<T>());
    gt_boxes_slice.Truncate(boxes_num);
    // Assign labels (-1, 0, 1) to all anchors
    AnchorLabelsAndMaxOverlapsInfo anchor_label_and_nearest_gt_box = AssignLabels(gt_boxes_slice, inside_anchors_slice, BnInOp2Blob); // TODO: refine AnchorLabelsAndMaxOverlapsInfo and GtBoxesNearestAnchorsInfo
    // Subsampe
    LabeledBBoxSlice<size_t, 3> labeled_anchor_slice();
    labeled_anchor_slice.GroupByLabelType();  // TODO
    int64_t start = 0;
    int64_t end = labeled_anchor_slice.;

    FOR_RANGE(int64_t, i, 0, labeled_anchor_slice.label_type_num()) {
      label = labeled_anchor_slice.label_ptr()[i];
      int32_t start_index = labeled_anchor_slice.get_label_start_index(label);
      int32_t count = labeled_anchor_slice.get_label_cnt(label);
      labeled_anchor_slice.shuffle(start_index, start_index + count);
    }
    const AnchorTargetOpConf& anchor_target_conf = GetCustomizedOpConf().anchors_target_conf();
    int32_t train_piece_size = anchor_target_conf.train_piece_size;
    int32_t fg_fraction = anchor_target_conf.fg_ratio;
    int32_t default_fg_cnt = train_piece_size * fg_ratio;

    fg_cnt = labeled_anchor_slice.get_label_cnt(1);
    bg_cnt = labeled_anchor_slice.get_label_cnt(0);
    // fg subsample
    if(fg_cnt > default_fg_cnt ) {
      fg_start = labeled_anchor_slice.get_label_start_index(1);
      FOR_RANGE(int32_t, i, fg_start, fg_cnt - default_fg_cnt) {
        labeled_anchor_slice.label_ptr[i] = -1;
      }
    } else {
      int32_t default_bg_cnt = train_piece_size - fg_cnt;
      bg_cnt <= default_bg_cnt ? bg_cnt : default_bg_cnt;
    }
    // bg subsample
    bg_start = labeled_anchor_slice.get_label_start_index(0);
    FOR_RANGE(int32_t, i, bg_start, bg_cnt) {
      labeled_anchor_slice.label_ptr[i] = -1;
    }
  }
}

/*
template<typename T>
struct AnchorTargetKernelUtil {
  static int32_t FindInsideAnchors(int32_t image_h, int32_t image_w, int32_t anchors_num,
                                   const T* anchors_ptr, int32_t* inside_anchors_inds_ptr) {
    int32_t cnt = 0;
    const BBox<T>* anchors_bbox = BBox<T>::Cast(anchors_ptr);
    FOR_RANGE(int32_t, i, 0, anchors_num) {
      if (anchors_bbox[i].x1() >= 0 && anchors_bbox[i].y1() >= 0 && anchors_bbox[i].x2() < image_w
          && anchors_bbox[i].y2() < image_h) {
        inside_anchors_inds_ptr[cnt++] = i;
      }
    }
    return cnt;
  }

  static void SetValue(int32_t size, int32_t* data_ptr, int32_t value) {
    FOR_RANGE(int32_t, i, 0, size) { data_ptr[i] = value; }
  }

  static void AssignPositiveLabel4GtMaxOverlap(int32_t anchors_num, int32_t gt_boxes_num,
                                               const int32_t* gt_max_overlaps_inds_ptr,
                                               const int32_t* gt_max_overlaps_num_ptr,
                                               int32_t* labels_ptr) {
    FOR_RANGE(int32_t, i, 0, gt_boxes_num) {
      FOR_RANGE(int32_t, j, 0, gt_max_overlaps_num_ptr[i]) {
        int32_t anchor_idx = gt_max_overlaps_inds_ptr[i * anchors_num + j];  // fix it
        labels_ptr[anchor_idx] = 1;
      }
    }
  }

  static void AssignLableByThreshold(int32_t inside_anchors_num,
                                     const int32_t* inside_anchors_inds_ptr,
                                     const float* max_overlaps_ptr, float high_threshold,
                                     float low_threshold, int32_t* labels_ptr) {
    FOR_RANGE(int32_t, i, 0, inside_anchors_num) {
      int32_t anchor_idx = inside_anchors_inds_ptr[i];
      const float overlap = max_overlaps_ptr[anchor_idx];
      if (overlap > high_threshold) {
        labels_ptr[anchor_idx] = 1;
      } else if (overlap < low_threshold) {
        labels_ptr[anchor_idx] = 0;
      }
    }
  }
};

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {

  
  const Blob* image_info_blob = BnInOp2Blob("image_info");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");

  Blob* rpn_labels_blob = BnInOp2Blob("rpn_labels");
  Blob* rpn_bbox_targets_blob = BnInOp2Blob("rpn_bbox_targets");
  Blob* rpn_bbox_inside_weights_blob = BnInOp2Blob("rpn_bbox_inside_weights");
  Blob* rpn_bbox_outside_weights_blob = BnInOp2Blob("rpn_bbox_outside_weights");

  const T* anchors_ptr = BnInOp2Blob("anchors")->dptr<T>();
  int32_t* inside_anchors_inds_ptr = BnInOp2Blob("inside_anchors_inds")->mut_dptr<int32_t>();
  int32_t* fg_inds_ptr = BnInOp2Blob("fg_inds")->mut_dptr<int32_t>();
  int32_t* bg_inds_ptr = BnInOp2Blob("bg_inds")->mut_dptr<int32_t>();
  float* max_overlaps_ptr = BnInOp2Blob("max_overlaps")->mut_dptr<float>();
  int32_t* max_overlaps_inds_ptr = BnInOp2Blob("max_overlaps_inds")->mut_dptr<int32_t>();
  int32_t* gt_max_overlaps_inds_ptr = BnInOp2Blob("gt_max_overlaps_inds")->mut_dptr<int32_t>();
  int32_t* gt_max_overlaps_num_ptr = BnInOp2Blob("gt_max_overlaps_num")->mut_dptr<int32_t>();
  T* origin_gt_boxes_ptr = BnInOp2Blob("origin_gt_boxes")->mut_dptr<T>();

  // useful vars
  const int32_t image_num = image_info_blob->shape().At(0);        // N
  const int32_t anchors_num = rpn_labels_blob->shape().Count(1);   // H*W*A
  const int32_t labels_num = rpn_labels_blob->shape().elem_cnt();  // N*H*W*A
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const int32_t image_height = conf.anchors_generator_conf().image_height();
  const int32_t image_width = conf.anchors_generator_conf().image_width();
  const BBoxRegressionWeights& bbox_reg_ws = conf.bbox_reg_weights();

  AnchorTargetKernelUtil<T>::SetValue(labels_num, rpn_labels_blob->mut_dptr<int32_t>(), -1);
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_inside_weights_blob->mut_dptr(), 0,
                           rpn_bbox_inside_weights_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_outside_weights_blob->mut_dptr(), 0,
                           rpn_bbox_outside_weights_blob->ByteSizeOfDataContentField());

  FOR_RANGE(int32_t, image_inds, 0, image_num) {  // for each image

    const BBox<T>* anchors_bbox = BBox<T>::Cast(anchors_ptr);
    const FloatList16* gt_boxes_ptr =
        gt_boxes_blob->dptr<FloatList16>() + image_inds;  // todo: fix it

    FOR_RANGE(int32_t, i, 0, gt_boxes_ptr->value().value_size()) {
      origin_gt_boxes_ptr[i] = gt_boxes_ptr->value().value(i) * 720;
    }

    const BBox<T>* current_img_gt_boxes_bbox = BBox<T>::Cast(origin_gt_boxes_ptr);
    const int32_t current_img_gt_boxes_num = gt_boxes_ptr->value().value_size() / 4;

    int32_t* current_img_label_ptr = rpn_labels_blob->mut_dptr<int32_t>(image_inds);
    BBoxDelta<T>* current_img_target_bbox_delta =
        BBoxDelta<T>::MutCast(rpn_bbox_targets_blob->mut_dptr<T>(image_inds));
    BBox<T>* current_img_inside_weights_bbox =
        BBox<T>::MutCast(rpn_bbox_inside_weights_blob->mut_dptr<T>(image_inds));
    BBox<T>* current_img_outside_weights_bbox =
        BBox<T>::MutCast(rpn_bbox_outside_weights_blob->mut_dptr<T>(image_inds));

    // 1. Find inside anchors.
    const int32_t inside_anchors_num = AnchorTargetKernelUtil<T>::FindInsideAnchors(
        image_height, image_width, anchors_num, anchors_ptr, inside_anchors_inds_ptr);

    // 2. Compute max_overlaps.
    FOR_RANGE(int32_t, i, 0, current_img_gt_boxes_num) {  // for each groundtruth box

      float gt_max_overlap = 0.0;
      int32_t gt_max_overlap_anchor_cnt = 0;

      FOR_RANGE(int32_t, j, 0, inside_anchors_num) {  // for each anchor
        int32_t anchor_idx = inside_anchors_inds_ptr[j];
        float overlap = anchors_bbox[anchor_idx].InterOverUnion(current_img_gt_boxes_bbox + i);
        if (overlap > max_overlaps_ptr[anchor_idx]) {
          max_overlaps_ptr[anchor_idx] = overlap;
          max_overlaps_inds_ptr[anchor_idx] = i;
        }

        if (overlap > gt_max_overlap) {
          gt_max_overlap = overlap;
          gt_max_overlaps_inds_ptr[i * anchors_num] = anchor_idx;
          gt_max_overlap_anchor_cnt = 1;
        } else if (overlap && overlap == gt_max_overlap) {
          gt_max_overlaps_inds_ptr[i * anchors_num + gt_max_overlap_anchor_cnt] = anchor_idx;
          gt_max_overlap_anchor_cnt++;
        }
        gt_max_overlaps_num_ptr[i] = gt_max_overlap_anchor_cnt;
      }
    }

    // 3. Label anchors, 1 is positive, 0 is negative, -1 is dont care.
    AnchorTargetKernelUtil<T>::AssignPositiveLabel4GtMaxOverlap(
        anchors_num, current_img_gt_boxes_num, gt_max_overlaps_inds_ptr, gt_max_overlaps_num_ptr,
        current_img_label_ptr);
    AnchorTargetKernelUtil<T>::AssignLableByThreshold(
        inside_anchors_num, inside_anchors_inds_ptr, max_overlaps_ptr,
        conf.positive_overlap_threshold(), conf.negative_overlap_threshold(),
        current_img_label_ptr);

    // 4. Subsample if needed.
    int32_t fg_cnt = 0;
    int32_t bg_cnt = 0;
    FOR_RANGE(int32_t, i, 0, inside_anchors_num) {
      int32_t anchor_idx = inside_anchors_inds_ptr[i];
      if (current_img_label_ptr[anchor_idx] == 1) {
        fg_inds_ptr[fg_cnt++] = anchor_idx;
      } else if (current_img_label_ptr[anchor_idx] == 0) {
        bg_inds_ptr[bg_cnt++] = anchor_idx;
      }
    }

    const int32_t fg_conf_size = conf.batchsize() * conf.foreground_fraction();
    if (fg_cnt > fg_conf_size) {
      // subsample fg
      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(fg_inds_ptr, fg_inds_ptr + fg_cnt, gen);
      fg_cnt = fg_conf_size;
    }
    const int32_t bg_conf_size = conf.batchsize() - fg_cnt;

    if (bg_cnt > bg_conf_size) {
      // subsampel bg
      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(bg_inds_ptr, bg_inds_ptr + bg_cnt, gen);
      bg_cnt = bg_conf_size;
    }
    LOG(INFO) << "fg_cnt(2): " << fg_cnt << std::endl;
    LOG(INFO) << "bg_cnt(2): " << bg_cnt << std::endl;

    // 5. Compute foreground anchors' bounding box regresion target(i.e. deltas).
    FOR_RANGE(int32_t, i, 0, fg_cnt) {z
      const int32_t anchor_idx = fg_inds_ptr[i];
      const int32_t gt_boxes_idx = max_overlaps_inds_ptr[anchor_idx];
      current_img_target_bbox_delta[anchor_idx].TransformInverse(
          anchors_bbox + anchor_idx, current_img_gt_boxes_bbox + gt_boxes_idx, bbox_reg_ws);
    }

    // 6. Set bounding box inside weights and outside weights.
    const float weight_value = 1.0 / (fg_cnt + bg_cnt);
    FOR_RANGE(int32_t, i, 0, fg_cnt) {
      const int32_t anchor_idx = fg_inds_ptr[i];
      current_img_inside_weights_bbox[anchor_idx].set_x1(1.0);
      current_img_inside_weights_bbox[anchor_idx].set_y1(1.0);
      current_img_inside_weights_bbox[anchor_idx].set_x2(1.0);
      current_img_inside_weights_bbox[anchor_idx].set_y2(1.0);

      current_img_outside_weights_bbox[anchor_idx].set_x1(weight_value);
      current_img_outside_weights_bbox[anchor_idx].set_y1(weight_value);
      current_img_outside_weights_bbox[anchor_idx].set_x2(weight_value);
      current_img_outside_weights_bbox[anchor_idx].set_y2(weight_value);
    }

    FOR_RANGE(int32_t, i, 0, bg_cnt) {
      const int32_t anchor_idx = bg_inds_ptr[i];
      current_img_outside_weights_bbox[anchor_idx].set_x1(weight_value);
      current_img_outside_weights_bbox[anchor_idx].set_y1(weight_value);
      current_img_outside_weights_bbox[anchor_idx].set_x2(weight_value);
      current_img_outside_weights_bbox[anchor_idx].set_y2(weight_value);
    }
  }
}
*/


ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
