#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

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
        int32_t anchor_idx = gt_max_overlaps_inds_ptr[i * anchors_num + j];
        labels_ptr[anchor_idx] = 1;
      }
    }
  }

  static void AssignLableByHighThreshold(int32_t inside_anchors_num,
                                         const int32_t* inside_anchors_inds_ptr,
                                         const float* max_overlaps_ptr, float high_threshold,
                                         int32_t* labels_ptr) {
    FOR_RANGE(int32_t, i, 0, inside_anchors_num) {
      int32_t anchor_idx = inside_anchors_inds_ptr[i];
      const float overlap = max_overlaps_ptr[anchor_idx];
      if (overlap >= high_threshold) { labels_ptr[anchor_idx] = 1; }
    }
  }
  static int32_t FindLableByLowThreshold(int32_t inside_anchors_num,
                                         const int32_t* inside_anchors_inds_ptr,
                                         const float* max_overlaps_ptr, float low_threshold,
                                         int32_t* bg_inds_ptr) {
    int32_t bg_cnt = 0;
    FOR_RANGE(int32_t, i, 0, inside_anchors_num) {
      int32_t anchor_idx = inside_anchors_inds_ptr[i];
      const float overlap = max_overlaps_ptr[anchor_idx];
      if (overlap < low_threshold) { bg_inds_ptr[bg_cnt++] = anchor_idx; }
    }
    return bg_cnt;
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
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_targets_blob->mut_dptr(), 0,
                           rpn_bbox_targets_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_inside_weights_blob->mut_dptr(), 0,
                           rpn_bbox_inside_weights_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_outside_weights_blob->mut_dptr(), 0,
                           rpn_bbox_outside_weights_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, max_overlaps_ptr, 0,
                           BnInOp2Blob("max_overlaps")->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, max_overlaps_inds_ptr, 0,
                           BnInOp2Blob("max_overlaps_inds")->ByteSizeOfDataContentField());
  int32_t debug_img_cnt = 0;
  FOR_RANGE(int32_t, image_inds, 0, image_num) {  // for each image

    const BBox<T>* anchors_bbox = BBox<T>::Cast(anchors_ptr);
    const FloatList16* gt_boxes_ptr =
        gt_boxes_blob->dptr<FloatList16>() + image_inds;  // todo: fix it

    FOR_RANGE(int32_t, i, 0, gt_boxes_ptr->value().value_size()) {
      origin_gt_boxes_ptr[i] = gt_boxes_ptr->value().value(i) * 720;
      if (i % 4 == 2 || i % 4 == 3) { origin_gt_boxes_ptr[i]--; }
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
        } else if (std::abs(overlap - gt_max_overlap) <= std::numeric_limits<float>::epsilon()) {
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
    AnchorTargetKernelUtil<T>::AssignLableByHighThreshold(
        inside_anchors_num, inside_anchors_inds_ptr, max_overlaps_ptr,
        conf.positive_overlap_threshold(), current_img_label_ptr);

    // Subsample fg if needed.
    int32_t fg_cnt = 0;
    FOR_RANGE(int32_t, i, 0, inside_anchors_num) {
      int32_t anchor_idx = inside_anchors_inds_ptr[i];
      if (current_img_label_ptr[anchor_idx] == 1) { fg_inds_ptr[fg_cnt++] = anchor_idx; }
    }

    const int32_t fg_conf_size = conf.batchsize() * conf.foreground_fraction();
    if (fg_cnt > fg_conf_size) {
      // std::random_device rd;
      // std::mt19937 gen(rd());
      // std::shuffle(fg_inds_ptr, fg_inds_ptr + fg_cnt, gen);
      // write back
      FOR_RANGE(int32_t, i, fg_conf_size, fg_cnt) { current_img_label_ptr[fg_inds_ptr[i]] = -1; }
      fg_cnt = fg_conf_size;
    }
    const int32_t bg_conf_size = conf.batchsize() - fg_cnt;
    int32_t bg_cnt = AnchorTargetKernelUtil<T>::FindLableByLowThreshold(
        inside_anchors_num, inside_anchors_inds_ptr, max_overlaps_ptr,
        conf.negative_overlap_threshold(), bg_inds_ptr);

    if (bg_cnt > bg_conf_size) {
      // subsampel bg
      // std::random_device rd;
      // std::mt19937 gen(rd());
      // std::shuffle(bg_inds_ptr, bg_inds_ptr + bg_cnt, gen);

      // write back
      FOR_RANGE(int32_t, i, 0, bg_conf_size) { current_img_label_ptr[bg_inds_ptr[i]] = 0; }
      bg_cnt = bg_conf_size;
    }
    LOG(INFO) << "fg_cnt(2): " << fg_cnt << std::endl;
    LOG(INFO) << "bg_cnt(2): " << bg_cnt << std::endl;

    // 5. Compute foreground anchors' bounding box regresion target(i.e. deltas).
    FOR_RANGE(int32_t, i, 0, fg_cnt) {
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
    debug_img_cnt++;
  }
}

template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FasterRcnnUtil<T>::GenerateAnchors(op_conf().anchor_target_conf().anchors_generator_conf(),
                                     BnInOp2Blob("anchors"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
