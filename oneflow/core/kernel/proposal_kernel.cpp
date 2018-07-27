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
      base_anchors[i * ratios_size * 4 + j * 4 + 0] = base_ctr - 0.5 * (wr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 1] = base_ctr - 0.5 * (hr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 2] = base_ctr + 0.5 * (wr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 3] = base_ctr + 0.5 * (hr - 1);
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
inline bool LtMinSize(T min_size, const T* bbox) {
  return (bbox[2] - bbox[0] < min_size) || (bbox[3] - bbox[1] < min_size);
}

}  // namespace

template<typename T>
void ProposalKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // 1. take fg channel of score
  // 2. bbox_transform_inv
  // 3. clip_boxes
  // 4. filter_boxes
  // 5. sort
  // 6. pre_nms_topN
  // 7. nms
  // 8. post_nms_topN
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  const Blob* class_prob_blob = BnInOp2Blob("class_prob");
  // const Blob* im_info_blob = BnInOp2Blob("image_info");
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* fg_prob_blob = nullptr;
  Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* keep_blob = BnInOp2Blob("keep");
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* roi_probs_blob = BnInOp2Blob("roi_probs");

  int64_t num_batch = class_prob_blob->shape().At(0);
  int64_t height = class_prob_blob->shape().At(1);
  int64_t width = class_prob_blob->shape().At(2);
  const ProposalOpConf& conf = op_conf().proposal_conf();
  int64_t num_anchors = conf.aspect_ratios_size() * conf.anchor_scales_size();
  int64_t num_proposals = height * width * num_anchors;
  bool input_only_fg_prob = conf.only_foreground_prob();

  if (input_only_fg_prob) {
    fg_prob_blob = const_cast<Blob*>(class_prob_blob);
  } else {
    fg_prob_blob = BnInOp2Blob("fg_prob");
    ProposalKernelUtil<DeviceType::kCPU, T>::TakeFgProbs(ctx.device_ctx, num_batch * height * width,
                                                         num_anchors, class_prob_blob->dptr<T>(),
                                                         fg_prob_blob->mut_dptr<T>());
  }
  FOR_RANGE(int64_t, i, 0, num_batch) {
    ProposalKernelUtil<DeviceType::kCPU, T>::BboxTransformInv(
        ctx.device_ctx, num_proposals, anchors_blob->dptr<T>(),
        bbox_pred_blob->dptr<T>() + i * num_proposals * 4,
        proposals_blob->mut_dptr<T>() + i * num_proposals * 4);
    // T image_height = im_info_blob->dptr<T>()[i * num_proposals];
    // T image_width = im_info_blob->dptr<T>()[i * num_proposals + 1];
    // T scale = im_info_blob->dptr<T>()[i * num_proposals + 2];
    T image_height = static_cast<T>(conf.image_height());
    T image_width = static_cast<T>(conf.image_width());
    T scale = static_cast<T>(1.0f);
    ProposalKernelUtil<DeviceType::kCPU, T>::ClipBoxes(
        ctx.device_ctx, i, num_proposals, image_height, image_width, proposals_blob->mut_dptr<T>());
    ProposalKernelUtil<DeviceType::kCPU, T>::FilterBoxesByMinSize(
        ctx.device_ctx, i, num_proposals, conf.min_size(), scale, keep_blob->mut_dptr<int32_t>(),
        proposals_blob->mut_dptr<T>());
    ProposalKernelUtil<DeviceType::kCPU, T>::SortByScore(
        ctx.device_ctx, i, num_proposals, keep_blob->dptr<int32_t>(), fg_prob_blob->dptr<T>(),
        proposals_blob->mut_dptr<T>());
    ProposalKernelUtil<DeviceType::kCPU, T>::Nms(
        ctx.device_ctx, i, num_proposals, conf.pre_nms_top_n(), conf.post_nms_top_n(),
        conf.nms_threshold(), proposals_blob->dptr<T>(), fg_prob_blob->dptr<T>(),
        keep_blob->dptr<int32_t>(), rois_blob->mut_dptr<T>(), roi_probs_blob->mut_dptr<T>());
  }
}

template<typename T>
void ProposalKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GenerateAnchors<T>(op_conf().proposal_conf(), BnInOp2Blob("anchors"));
}

template<typename T>
struct ProposalKernelUtil<DeviceType::kCPU, T> {
  static void TakeFgProbs(DeviceCtx* ctx, int64_t m, int64_t a, const T* class_prob, T* fg_prob) {
    // m = n * h * w
    // class_prob (n, h, w, a * 2)
    // fg_prob (n, h, w, a)
    UNIMPLEMENTED();
  }

  static void BboxTransform(DeviceCtx* ctx, int64_t m, const T* bbox, const T* target_bbox,
                            T* deltas) {
    // m = h * w * a
    // shape: bbox == target_bbox == deltas (h, w, a * 4)
    FOR_RANGE(int64_t, i, 0, m) {
      float b_w = bbox[i * 4 + 2] - bbox[i * 4] + 1.0f;
      float b_h = bbox[i * 4 + 3] - bbox[i * 4 + 1] + 1.0f;
      float b_ctr_x = bbox[i * 4] + 0.5f * b_w;
      float b_ctr_y = bbox[i * 4 + 1] + 0.5f * b_h;

      float t_w = target_bbox[i * 4 + 2] - target_bbox[i * 4] + 1.0f;
      float t_h = target_bbox[i * 4 + 3] - target_bbox[i * 4 + 1] + 1.0f;
      float t_ctr_x = target_bbox[i * 4] + 0.5f * t_w;
      float t_ctr_y = target_bbox[i * 4 + 1] + 0.5f * t_h;

      deltas[i * 4 + 0] = (t_ctr_x - b_ctr_x) / b_w;
      deltas[i * 4 + 1] = (t_ctr_y - b_ctr_y) / b_h;
      deltas[i * 4 + 2] = std::log(t_w / b_w);
      deltas[i * 4 + 3] = std::log(t_h / b_h);
    }
  }

  static void BboxTransformInv(DeviceCtx* ctx, int64_t m, const T* bbox, const T* deltas,
                               T* bbox_pred) {
    // m = h * w * a
    // bbox == deltas == bbox_pred (h * w * a * 4)
    FOR_RANGE(int64_t, i, 0, m) {
      float w = bbox[i * 4 + 2] - bbox[i * 4 + 0] + 1.0f;
      float h = bbox[i * 4 + 3] - bbox[i * 4 + 1] + 1.0f;
      float ctr_x = bbox[i * 4 + 0] + 0.5f * w;
      float ctr_y = bbox[i * 4 + 1] + 0.5f * h;

      float pred_ctr_x = deltas[i * 4 + 0] * w + ctr_x;
      float pred_ctr_y = deltas[i * 4 + 1] * h + ctr_y;
      float pred_w = std::exp(deltas[i * 4 + 2]) * w;
      float pred_h = std::exp(deltas[i * 4 + 3]) * h;

      bbox_pred[i * 4 + 0] = pred_ctr_x - 0.5f * pred_w;
      bbox_pred[i * 4 + 1] = pred_ctr_y - 0.5f * pred_h;
      bbox_pred[i * 4 + 2] = pred_ctr_x + 0.5f * pred_w - 1.f;
      bbox_pred[i * 4 + 3] = pred_ctr_y + 0.5f * pred_h - 1.f;
    }
  }

  static void ClipBoxes(DeviceCtx* ctx, int64_t index, int64_t num_proposals, T image_height,
                        T image_width, T* proposals) {
    // proposals (n, h, w, a * 4)
    FOR_RANGE(int64_t, i, 0, num_proposals) {
      int64_t cur = (index * num_proposals + i) * 4;
      // x1 >= 0 && y1 >= 0 && x2 <= origin_width && y2 <= origin_height
      proposals[cur + 0] = std::max(std::min(proposals[cur + 0], image_width), static_cast<T>(0));
      proposals[cur + 1] = std::max(std::min(proposals[cur + 1], image_height), static_cast<T>(0));
      proposals[cur + 2] = std::max(std::min(proposals[cur + 2], image_width), static_cast<T>(0));
      proposals[cur + 3] = std::max(std::min(proposals[cur + 3], image_height), static_cast<T>(0));
    }
  }

  static void FilterBoxesByMinSize(DeviceCtx* ctx, int64_t batch_index, int64_t num_proposals,
                                   int32_t min_size, T scale, int32_t* keep, T* proposals) {
    // m = h * w * a
    T real_min_size = static_cast<T>(min_size * scale);
    T* cur_img_proposals = proposals + batch_index * num_proposals * 4;
    T* drop_ptr = cur_img_proposals;
    T* keep_ptr = cur_img_proposals + (num_proposals - 1) * 4;
    while (drop_ptr <= keep_ptr) {
      while (drop_ptr <= keep_ptr && !LtMinSize(real_min_size, drop_ptr)) { drop_ptr += 4; }
      while (drop_ptr <= keep_ptr && LtMinSize(real_min_size, keep_ptr)) { keep_ptr -= 4; }
      if (drop_ptr < keep_ptr) { std::swap_ranges(drop_ptr, drop_ptr + 4, keep_ptr); }
    }
    keep[batch_index] = (drop_ptr - cur_img_proposals) / 4;
  }

  static void SortByScore(DeviceCtx* ctx, int64_t index, int64_t num_proposals, const int32_t* keep,
                          const T* scores, T* proposals) {
    // scores (n, h * w * a, 1)
    // proposals (n, h * w * a, 4)
    int32_t keep_to = keep[index];
    T* cur_scores = const_cast<T*>(scores + index * num_proposals);
    T* cur_proposals = const_cast<T*>(proposals + index * num_proposals * 4);
    std::vector<size_t> sort_indexes(keep_to);
    std::iota(sort_indexes.begin(), sort_indexes.end(), 0);
    std::sort(sort_indexes.begin(), sort_indexes.end(),
              [&](size_t idx1, size_t idx2) { return cur_scores[idx1] > cur_scores[idx2]; });
    FOR_RANGE(int64_t, i, 0, sort_indexes.size()) {
      int64_t idx = i;
      int64_t val = sort_indexes[i];
      while (sort_indexes[idx] != val) {
        std::iter_swap(cur_scores + idx, cur_scores + val);
        std::swap_ranges(cur_proposals + idx * 4, cur_proposals + (idx + 1) * 4,
                         cur_proposals + val * 4);
        val = idx;
        idx = std::find(sort_indexes.begin(), sort_indexes.end(), idx) - sort_indexes.begin();
      }
    }
  }

  static void Nms(DeviceCtx* ctx, int64_t index, int64_t num_proposals, int64_t pre_nms_top_n,
                  int64_t post_nms_top_n, float nms_threshold, const T* proposals, const T* probs,
                  const int32_t* keep, T* rois, T* roi_probs) {
    // proposals (n, h * w * a, 4)
    // probs (n, h * w * a, 1)
    // rois (n * post_nms_top_n, 5)
    // roi_probs (n * post_nms_top_n, 1)
    int64_t keep_to = static_cast<int64_t>(keep[index]);
    int64_t topn = pre_nms_top_n == -1 ? keep_to : std::min(keep_to, pre_nms_top_n);
    std::vector<bool> suppressed(topn, false);
    std::vector<int64_t> remain;
    FOR_RANGE(int64_t, i, 0, topn) {
      if (suppressed[i]) { continue; }
      remain.push_back(i);
      const T* cur_prop = proposals + (index * num_proposals + i) * 4;
      FOR_RANGE(int64_t, j, i + 1, topn) {
        const T* cand_prop = proposals + (index * num_proposals + j) * 4;
        float iou = InterOverUnion(cur_prop, cand_prop);
        // float iou = (*cur_prop) + (*cand_prop);
        if (iou >= nms_threshold) { suppressed[j] = true; }
      }
    }

    FOR_RANGE(int64_t, i, 0, std::min(remain.size(), static_cast<size_t>(post_nms_top_n))) {
      // rois[(index * post_nms_top_n + i) * 4] = index;
      FOR_RANGE(int64_t, j, 0, 4) {
        rois[(index * post_nms_top_n + i) * 4 + j] =
            proposals[(index * num_proposals + remain[i]) * 4 + j];
      }
      roi_probs[index * post_nms_top_n + i] = probs[index * num_proposals + remain[i]];
    }
  }

  inline static float BBoxArea(const T* box) {
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
  }

  inline static float InterOverUnion(const T* box1, const T* box2) {
    float iou = 0;
    int32_t iw = std::min(box1[2], box2[2]) - std::max(box1[0], box2[0]) + 1;
    if (iw > 0) {
      int32_t ih = std::min(box1[3], box2[3]) - std::max(box1[1], box2[1]) + 1;
      if (ih > 0) {
        float inter = iw * ih;
        float ua = BBoxArea(box1) + BBoxArea(box2) - inter;
        iou = inter / ua;
      }
    }
    return iou;
  }
};

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalConf, ProposalKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
