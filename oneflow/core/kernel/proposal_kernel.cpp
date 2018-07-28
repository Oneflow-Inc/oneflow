#include "oneflow/core/kernel/proposal_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
void GenerateAnchors(const ProposalOpConf& conf, Blob* anchors_blob) {
  // anchors_blob shape (H, W, A * 4)
  int32_t height = anchors_blob->shape().At(0);
  int32_t width = anchors_blob->shape().At(1);
  int32_t scales_size = conf.anchor_scales_size();
  int32_t ratios_size = conf.aspect_ratios_size();
  int32_t num_anchors = scales_size * ratios_size;
  CHECK_EQ(num_anchors * 4, anchors_blob->shape().At(2));
  int32_t fm_stride = conf.feature_map_stride();
  float base_ctr = 0.5 * (fm_stride - 1);

  std::vector<T> base_anchors(num_anchors * 4);
  FOR_RANGE(int32_t, i, 0, scales_size) {
    int32_t ws = conf.anchor_scales(i);
    int32_t hs = conf.anchor_scales(i);
    FOR_RANGE(int32_t, j, 0, ratios_size) {
      float wr = std::sqrt(hs * ws / conf.aspect_ratios(j));
      float hr = wr * conf.aspect_ratios(j);
      LOG(INFO) << "scaled ratio height: " << hr << ", width: " << wr;
      base_anchors[(i * ratios_size + j) * 4 + 0] = base_ctr - 0.5 * (wr - 1);
      base_anchors[(i * ratios_size + j) * 4 + 1] = base_ctr - 0.5 * (hr - 1);
      base_anchors[(i * ratios_size + j) * 4 + 2] = base_ctr + 0.5 * (wr - 1);
      base_anchors[(i * ratios_size + j) * 4 + 3] = base_ctr + 0.5 * (hr - 1);
    }
  }

  T* anchors_dptr = anchors_blob->mut_dptr<T>();
  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      int32_t cur = (h * width + w) * num_anchors * 4;
      FOR_RANGE(int32_t, i, 0, num_anchors) {
        anchors_dptr[cur + i * 4 + 0] = base_anchors[i * 4 + 0] + w * fm_stride;
        anchors_dptr[cur + i * 4 + 1] = base_anchors[i * 4 + 1] + h * fm_stride;
        anchors_dptr[cur + i * 4 + 2] = base_anchors[i * 4 + 2] + w * fm_stride;
        anchors_dptr[cur + i * 4 + 3] = base_anchors[i * 4 + 3] + h * fm_stride;
      }
    }
  }
}

template<typename T>
inline bool LtMinSize(const int32_t min_size, const T* bbox) {
  return (bbox[2] - bbox[0] + 1 < min_size) || (bbox[3] - bbox[1] + 1 < min_size);
}

template<typename T>
inline float BBoxArea(const T* box) {
  return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
}

template<typename T>
inline float InterOverUnion(const T* box0, const T* box1) {
  float iou = 0;
  int32_t iw = std::min(box0[2], box1[2]) - std::max(box0[0], box1[0]) + 1;
  iw = std::max<int32_t>(iw, 0);
  int32_t ih = std::min(box0[3], box1[3]) - std::max(box0[1], box1[1]) + 1;
  ih = std::max<int32_t>(ih, 0);
  float inter = iw * ih;
  float ua = BBoxArea(box0) + BBoxArea(box1) - inter;
  iou = inter / ua;
  return iou;
}

}  // namespace

template<typename T>
struct ProposalKernelUtil {
  static void CopyRoI(const int32_t* sorted_score_index_ptr, const int32_t post_nms_num,
                      const T* const_img_proposal_ptr, T* rois_ptr) {
    FOR_RANGE(int32_t, roi_idx, 0, post_nms_num) {
      int32_t roi_offset = roi_idx * 4;
      int32_t proposal_offset = sorted_score_index_ptr[roi_idx] * 4;
      rois_ptr[roi_offset + 0] = const_img_proposal_ptr[proposal_offset + 0];
      rois_ptr[roi_offset + 1] = const_img_proposal_ptr[proposal_offset + 1];
      rois_ptr[roi_offset + 2] = const_img_proposal_ptr[proposal_offset + 2];
      rois_ptr[roi_offset + 3] = const_img_proposal_ptr[proposal_offset + 3];
    }
  }

  static void BboxTransformInv(int64_t img_proposal_num, const T* bbox, const T* target_bbox,
                               T* deltas) {
    // img_proposal_num = h * w * a
    // boxes: (h, w, a, 4)
    int64_t bboxes_buf_size = img_proposal_num * 4;
    for (int64_t i = 0; i < bboxes_buf_size; i += 4) {
      float b_w = bbox[i + 2] - bbox[i + 0] + 1.0f;
      float b_h = bbox[i + 3] - bbox[i + 1] + 1.0f;
      float b_ctr_x = bbox[i + 0] + 0.5f * b_w;
      float b_ctr_y = bbox[i + 1] + 0.5f * b_h;

      float t_w = target_bbox[i + 2] - target_bbox[i + 0] + 1.0f;
      float t_h = target_bbox[i + 3] - target_bbox[i + 1] + 1.0f;
      float t_ctr_x = target_bbox[i + 0] + 0.5f * t_w;
      float t_ctr_y = target_bbox[i + 1] + 0.5f * t_h;

      deltas[i + 0] = (t_ctr_x - b_ctr_x) / b_w;
      deltas[i + 1] = (t_ctr_y - b_ctr_y) / b_h;
      deltas[i + 2] = std::log(t_w / b_w);
      deltas[i + 3] = std::log(t_h / b_h);
    }
  }

  static void BboxTransform(int64_t img_proposal_num, const T* boxes, const T* deltas,
                            T* bbox_pred) {
    // img_proposal_num = h * w * a
    // boxes: (h, w, a, 4)
    int64_t bboxes_buf_size = img_proposal_num * 4;
    for (int64_t i = 0; i < bboxes_buf_size; i += 4) {
      float w = boxes[i + 2] - boxes[i + 0] + 1.0f;
      float h = boxes[i + 3] - boxes[i + 1] + 1.0f;
      float ctr_x = boxes[i + 0] + 0.5f * w;
      float ctr_y = boxes[i + 1] + 0.5f * h;

      float pred_ctr_x = deltas[i + 0] * w + ctr_x;
      float pred_ctr_y = deltas[i + 1] * h + ctr_y;
      float pred_w = std::exp(deltas[i + 2]) * w;
      float pred_h = std::exp(deltas[i + 3]) * h;

      bbox_pred[i + 0] = pred_ctr_x - 0.5f * pred_w;
      bbox_pred[i + 1] = pred_ctr_y - 0.5f * pred_h;
      bbox_pred[i + 2] = pred_ctr_x + 0.5f * pred_w - 1.f;
      bbox_pred[i + 3] = pred_ctr_y + 0.5f * pred_h - 1.f;
    }
  }

  static void ClipBoxes(int64_t img_proposal_num, const int64_t image_height,
                        const int64_t image_width, T* proposals_ptr) {
    // img_proposal_num = h * w * a
    // proposals_ptr: (h, w, a, 4)
    const int32_t proposal_buf_size = img_proposal_num * 4;
    for (int64_t i = 0; i < proposal_buf_size; i += 4) {
      proposals_ptr[i + 0] = std::max<T>(std::min<T>(proposals_ptr[i + 0], image_width), 0);
      proposals_ptr[i + 1] = std::max<T>(std::min<T>(proposals_ptr[i + 1], image_height), 0);
      proposals_ptr[i + 2] = std::max<T>(std::min<T>(proposals_ptr[i + 2], image_width), 0);
      proposals_ptr[i + 3] = std::max<T>(std::min<T>(proposals_ptr[i + 3], image_height), 0);
    }
  }

  static int32_t FilterBoxesByMinSize(const int64_t im_proposal_num, const int32_t min_size,
                                      const T* img_proposal_ptr, int32_t* sorted_score_index_ptr) {
    int32_t keep_num = 0;
    FOR_RANGE(int32_t, i, 0, im_proposal_num) {
      const int32_t index = sorted_score_index_ptr[i];
      const int32_t img_proposal_offset = index * 4;
      if (!LtMinSize(min_size, img_proposal_ptr + img_proposal_offset)) {
        sorted_score_index_ptr[keep_num++] = index;
      }
    }
    return keep_num;
  }

  static void SortScoreIndex(const int64_t img_proposal_num, const T* score_ptr,
                             int32_t* sorted_score_index_ptr) {
    FOR_RANGE(int64_t, i, 0, img_proposal_num) { sorted_score_index_ptr[i] = i; }
    std::sort(sorted_score_index_ptr, sorted_score_index_ptr + img_proposal_num,
              [&](int32_t lhs, int32_t rhs) { return score_ptr[lhs] > score_ptr[rhs]; });
  }

  static int32_t Nms(const T* const_img_proposal_ptr, const float nms_threshold,
                     int32_t* sorted_score_index_ptr, const int32_t pre_nms_num,
                     int32_t* supressed_index_ptr) {
    FOR_RANGE(int32_t, i, 0, pre_nms_num) { supressed_index_ptr[i] = false; }
    int32_t keep_num = 0;
    FOR_RANGE(int32_t, i, 0, pre_nms_num) {
      if (supressed_index_ptr[i]) { continue; }
      sorted_score_index_ptr[keep_num++] = sorted_score_index_ptr[i];
      const int32_t cur_proposal_offset = sorted_score_index_ptr[i] * 4;
      const T* cur_proposal_ptr = const_img_proposal_ptr + cur_proposal_offset;
      FOR_RANGE(int32_t, j, i + 1, pre_nms_num) {
        if (supressed_index_ptr[j]) { continue; }
        const int32_t cand_proposal_offset = sorted_score_index_ptr[j] * 4;
        const T* cand_proposal_ptr = const_img_proposal_ptr + cand_proposal_offset;
        if (InterOverUnion(cur_proposal_ptr, cand_proposal_ptr) >= nms_threshold) {
          supressed_index_ptr[j] = true;
        }
      }
    }
    return keep_num;
  }
};

template<typename T>
void ProposalKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  const Blob* class_prob_blob = BnInOp2Blob("class_prob");
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* roi_probs_blob = BnInOp2Blob("roi_probs");
  int32_t* sorted_score_index_ptr = BnInOp2Blob("sorted_score_index")->mut_dptr<int32_t>();
  int32_t* supressed_index_ptr = BnInOp2Blob("suppressed_index")->mut_dptr<int32_t>();
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const int64_t height = class_prob_blob->shape().At(1);
  const int64_t width = class_prob_blob->shape().At(2);
  const int64_t num_anchors = conf.aspect_ratios_size() * conf.anchor_scales_size();

  const int64_t img_proposal_num = height * width * num_anchors;
  FOR_RANGE(int64_t, i, 0, class_prob_blob->shape().At(0)) {
    const int32_t img_score_offset = i * img_proposal_num;
    const T* class_score_ptr = class_prob_blob->dptr<T>() + img_score_offset;
    ProposalKernelUtil<T>::SortScoreIndex(img_proposal_num, class_score_ptr,
                                          sorted_score_index_ptr);

    const int32_t img_proposal_offset = i * img_proposal_num * 4;
    const T* const_img_proposal_ptr = proposals_blob->dptr<T>() + img_proposal_offset;
    T* mut_img_proposal_ptr = proposals_blob->mut_dptr<T>() + img_proposal_offset;
    ProposalKernelUtil<T>::BboxTransform(img_proposal_num, anchors_blob->dptr<T>(),
                                         bbox_pred_blob->dptr<T>() + img_proposal_offset,
                                         mut_img_proposal_ptr);

    ProposalKernelUtil<T>::ClipBoxes(img_proposal_num, conf.image_height(), conf.image_width(),
                                     mut_img_proposal_ptr);

    int32_t keep_num = ProposalKernelUtil<T>::FilterBoxesByMinSize(
        img_proposal_num, conf.min_size(), const_img_proposal_ptr, sorted_score_index_ptr);

    int32_t pre_nms_num = keep_num;
    if (conf.pre_nms_top_n() > 0) { pre_nms_num = std::min<int32_t>(keep_num, pre_nms_num); }
    int32_t post_nms_num =
        ProposalKernelUtil<T>::Nms(const_img_proposal_ptr, conf.nms_threshold(),
                                   sorted_score_index_ptr, pre_nms_num, supressed_index_ptr);

    // use duplicated rois if post_nms_num < conf.post_nms_top_n()
    for (int32_t box_i = 0; post_nms_num < conf.post_nms_top_n(); ++box_i, ++post_nms_num) {
      sorted_score_index_ptr[post_nms_num] = sorted_score_index_ptr[box_i];
    }

    post_nms_num = std::min<int32_t>(post_nms_num, conf.post_nms_top_n());
    T* rois_ptr = rois_blob->mut_dptr<T>() + img_proposal_offset;
    ProposalKernelUtil<T>::CopyRoI(sorted_score_index_ptr, post_nms_num, const_img_proposal_ptr,
                                   rois_ptr);

    T* roi_probs_ptr = roi_probs_blob->mut_dptr<T>() + img_score_offset;
    FOR_RANGE(int32_t, rois_idx, 0, post_nms_num) {
      roi_probs_ptr[rois_idx] = class_score_ptr[sorted_score_index_ptr[rois_idx]];
    }
  }
}

template<typename T>
void ProposalKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GenerateAnchors<T>(op_conf().proposal_conf(), BnInOp2Blob("anchors"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalConf, ProposalKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
