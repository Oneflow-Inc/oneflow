#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
void FasterRcnnUtil<T>::GenerateAnchors(const AnchorGeneratorConf& conf, Blob* anchors_blob) {
  // anchors_blob shape (h, w, a, 4)
  const int32_t height = anchors_blob->shape().At(0);
  const int32_t width = anchors_blob->shape().At(1);
  const int32_t scales_size = conf.anchor_scales_size();
  const int32_t ratios_size = conf.aspect_ratios_size();
  const int32_t fm_stride = conf.feature_map_stride();
  const int32_t num_anchors = scales_size * ratios_size;
  CHECK_EQ(num_anchors, anchors_blob->shape().At(2));

  const float base_ctr = 0.5 * (fm_stride - 1);
  std::vector<T> base_anchors(num_anchors * 4);
  BBox<T>* base_anchor_bbox = BBox<T>::MutCast(base_anchors.data());
  FOR_RANGE(int32_t, i, 0, ratios_size) {
    const int32_t wr = std::round(std::sqrt(fm_stride * fm_stride / conf.aspect_ratios(i)));
    const int32_t hr = std::round(wr * conf.aspect_ratios(i));
    FOR_RANGE(int32_t, j, 0, scales_size) {
      const float scale = conf.anchor_scales(j) / fm_stride;
      const int32_t ws = wr * scale;
      const int32_t hs = hr * scale;
      BBox<T>* cur_anchor_bbox = base_anchor_bbox + i * scales_size + j;
      cur_anchor_bbox->set_x1(base_ctr - 0.5 * (ws - 1));
      cur_anchor_bbox->set_y1(base_ctr - 0.5 * (hs - 1));
      cur_anchor_bbox->set_x2(base_ctr + 0.5 * (ws - 1));
      cur_anchor_bbox->set_y2(base_ctr + 0.5 * (hs - 1));
    }
  }

  const BBox<T>* const_base_anchor_bbox = BBox<T>::Cast(base_anchors.data());
  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      BBox<T>* anchor_bbox = BBox<T>::MutCast(anchors_blob->mut_dptr<T>(h, w));
      FOR_RANGE(int32_t, i, 0, num_anchors) {
        anchor_bbox[i].set_x1(const_base_anchor_bbox[i].x1() + w * fm_stride);
        anchor_bbox[i].set_y1(const_base_anchor_bbox[i].y1() + h * fm_stride);
        anchor_bbox[i].set_x2(const_base_anchor_bbox[i].x2() + w * fm_stride);
        anchor_bbox[i].set_y2(const_base_anchor_bbox[i].y2() + h * fm_stride);
      }
    }
  }
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransform(int64_t boxes_num, const T* bboxes, const T* deltas,
                                      const BBoxRegressionWeights& bbox_reg_ws, T* pred_bboxes) {
  FOR_RANGE(int64_t, i, 0, boxes_num) {
    BBox<T>::MutCast(pred_bboxes)[i].Transform(BBox<T>::Cast(bboxes) + i,
                                               BBoxDelta<T>::Cast(deltas) + i, bbox_reg_ws);
  }
}

template<typename T>
void FasterRcnnUtil<T>::BboxTransformInv(int64_t boxes_num, const T* bboxes, const T* target_bboxes,
                                         const BBoxRegressionWeights& bbox_reg_ws, T* deltas) {
  FOR_RANGE(int64_t, i, 0, boxes_num) {
    BBoxDelta<T>::MutCast(deltas)[i].TransformInverse(
        BBox<T>::Cast(bboxes) + i, BBox<T>::Cast(target_bboxes) + i, bbox_reg_ws);
  }
}

template<typename T>
void FasterRcnnUtil<T>::ClipBoxes(int64_t boxes_num, const int64_t image_height,
                                  const int64_t image_width, T* bboxes) {
  BBox<T>* bbox_ptr = BBox<T>::MutCast(bboxes);
  FOR_RANGE(int64_t, i, 0, boxes_num) { bbox_ptr[i].Clip(image_height, image_width); }
}

template<typename T>
void FasterRcnnUtil<T>::Nms(float nms_threshold, const ScoredBoxesIndex<T>& pre_nms_boxes,
                            ScoredBoxesIndex<T>& post_nms_boxes) {
  CHECK_NE(pre_nms_boxes.index_ptr(), post_nms_boxes.index_ptr());
  CHECK_EQ(pre_nms_boxes.bbox_ptr(), post_nms_boxes.bbox_ptr());
  CHECK_EQ(pre_nms_boxes.score_ptr(), post_nms_boxes.score_ptr());

  size_t keep_num = 0;
  auto IsSuppressed = [&](size_t pre_nms_n) -> bool {
    const BBox<T>* cur_bbox = pre_nms_boxes.GetBBox(pre_nms_n);
    FOR_RANGE(size_t, post_nms_i, 0, keep_num) {
      const BBox<T>* keep_bbox = post_nms_boxes.GetBBox(post_nms_i);
      if (keep_bbox->InterOverUnion(cur_bbox) >= nms_threshold) { return true; }
    }
    return false;
  };
  FOR_RANGE(size_t, pre_nms_i, 0, pre_nms_boxes.size()) {
    if (IsSuppressed(pre_nms_i)) { continue; }
    post_nms_boxes.mut_index_ptr()[keep_num++] = pre_nms_boxes.GetIndex(pre_nms_i);
    if (keep_num == post_nms_boxes.size()) { break; }
  }
  post_nms_boxes.Truncate(keep_num);

  CHECK_LE(post_nms_boxes.size(), pre_nms_boxes.size());
}

template<typename T>
void FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
    const BoxesIndex<T>& boxes, const GtBoxes& gt_boxes,
    const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(int32_t, i, 0, gt_boxes.size()) {
    FOR_RANGE(int32_t, j, 0, boxes.size()) {
      float overlap = boxes.GetBBox(j)->InterOverUnion(gt_boxes.GetBBox<float>(i));
      Handler(boxes.GetIndex(j), i, overlap);
    }
  }
}

template<typename T>
void FasterRcnnUtil<T>::CorrectGtBoxCoord(int32_t im_h, int32_t im_w, BBox<float>* bbox) {
  CHECK_GE(bbox->x1(), 0.f);
  CHECK_GE(bbox->y1(), 0.f);
  CHECK_GE(bbox->x2(), 0.f);
  CHECK_GE(bbox->y2(), 0.f);
  CHECK_LE(bbox->x1(), im_w);
  CHECK_LE(bbox->y1(), im_h);
  CHECK_LE(bbox->x2(), im_w);
  CHECK_LE(bbox->y2(), im_h);

  static int32_t gt_box_been_norm = -1;
  if (bbox->x1() <= 1.f && bbox->y1() <= 1.f && bbox->x2() <= 1.f && bbox->y2() <= 1.f) {
    CHECK_NE(gt_box_been_norm, 0);
    gt_box_been_norm = 1;
  } else {
    CHECK_NE(gt_box_been_norm, 1);
    gt_box_been_norm = 0;
  }

  if (gt_box_been_norm == 1) {
    bbox->set_x1(bbox->x1() * im_w);
    bbox->set_y1(bbox->y1() * im_h);
    bbox->set_x2(bbox->x2() * im_w - 1);
    bbox->set_y2(bbox->y2() * im_h - 1);
  } else {
    bbox->set_x2(bbox->x2() - 1);
    bbox->set_y2(bbox->y2() - 1);
  }
}

#define INITIATE_FASTER_RCNN_UTIL(T, type_cpp) template struct FasterRcnnUtil<T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_FASTER_RCNN_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
