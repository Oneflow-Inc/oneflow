#include "oneflow/core/kernel/proposal_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

void GenerateAnchors(const ProposalOpConf& conf, Blob* anchors_blob) {
  // anchors_blob shape (H * W * A, 4)
  int32_t* anchors_dptr = anchors_blob->mut_dptr<int32_t>();
  int32_t height = anchors_blob->shape().At(1);
  int32_t width = anchors_blob->shape().At(2);
  int32_t scales_size = conf.anchor_scales_size();
  int32_t ratios_size = conf.aspect_ratios_size();
  int32_t num_anchors = scales_size * ratios_size;
  int32_t fm_stride = conf.feature_map_stride();
  float base_ctr = 0.5 * (fm_stride - 1);

  std::vector<int32_t> base_anchors(num_anchors * 4);
  FOR_RANGE(int32_t, i, 0, scales_size) {
    FOR_RANGE(int32_t, j, 0, ratios_size) {
      int32_t ws = width * conf.anchor_scales(i);
      int32_t hs = height * conf.anchor_scales(i);
      float wr = std::sqrt(hs * ws / conf.aspect_ratios(j));
      float hr = wr * conf.aspect_ratios(j);
      base_anchors[i * ratios_size * 4 + j * 4] = base_ctr - 0.5 * (wr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 1] = base_ctr - 0.5 * (hr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 2] = base_ctr + 0.5 * (wr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 3] = base_ctr + 0.5 * (hr - 1);
    }
  }

  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      int32_t* anchors = anchors_dptr + (h * width + w) * num_anchors * 4;
      FOR_RANGE(int32_t, i, 0, base_anchors.size()) {
        if (i % 2 == 0) {
          *(anchors + i) = base_anchors[i] + w * fm_stride;
        } else {
          *(anchors + i) = base_anchors[i] + h * fm_stride;
        }
      }
    }
  }
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
  const Blob* class_prob_blob = BnInOp2Blob("class_prob");
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  const Blob* im_info_blob = BnInOp2Blob("image_info");
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* fg_scores_blob = BnInOp2Blob("fg_scores");
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* scores_blob = BnInOp2Blob("scores");

  int64_t num_batch = class_prob_blob->shape().At(0);
  int64_t height = class_prob_blob->shape().At(1);
  int64_t width = class_prob_blob->shape().At(2);
  const ProposalOpConf& conf = op_conf().proposal_conf();
  int64_t num_anchors = conf.aspect_ratios_size() * conf.anchor_scales_size();

  ProposalKernelUtil<DeviceType::kCPU, T>::TakeFgScores(ctx.device_ctx, num_batch * height * width,
                                                        num_anchors, class_prob_blob->dptr<T>(),
                                                        fg_scores_blob->mut_dptr<T>());
  ProposalKernelUtil<DeviceType::kCPU, T>::BboxTransformInv(
      ctx.device_ctx, num_batch * height * width * num_anchors, anchors_blob->dptr<T>(),
      bbox_pred_blob->dptr<T>(), proposals_blob->mut_dptr<T>());
  ProposalKernelUtil<DeviceType::kCPU, T>::ClipBoxes(
      ctx.device_ctx, num_batch, height * width * num_anchors, im_info_blob->dptr<T>(),
      proposals_blob->mut_dptr<T>());
  std::vector<int64_t> keep_to = ProposalKernelUtil<DeviceType::kCPU, T>::FilterBoxesByMinSize(
      ctx.device_ctx, num_batch, height * width * num_anchors, conf.min_size(),
      im_info_blob->dptr<T>(), proposals_blob->mut_dptr<T>());
  ProposalKernelUtil<DeviceType::kCPU, T>::SortByScore(
      ctx.device_ctx, num_batch, height * width * num_anchors, keep_to,
      fg_scores_blob->mut_dptr<T>(), proposals_blob->mut_dptr<T>());
  ProposalKernelUtil<DeviceType::kCPU, T>::Nms(
      ctx.device_ctx, num_batch, height * width * num_anchors, keep_to, conf.pre_nms_top_n(),
      conf.post_nms_top_n(), conf.nms_threshold(), proposals_blob->dptr<T>(),
      fg_scores_blob->dptr<T>(), rois_blob->mut_dptr<T>(), scores_blob->mut_dptr<T>());
}

template<typename T>
void ProposalKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GenerateAnchors(op_conf().proposal_conf(), BnInOp2Blob("anchors"));
}

template<typename T>
struct ProposalKernelUtil<DeviceType::kCPU, T> {
  static void TakeFgScores(DeviceCtx* ctx, int64_t m, int64_t a, const T* class_prob,
                           T* fg_scores) {
    // m = n * h * w
    // class_prob (n, h, w, a * 2)
    // fg_scores (n, h * w * a, 1)
    for (int64_t i = 0; i < m * a; ++i) { fg_scores[i] = class_prob[i / m * a * 2 + i % m + a]; }
  }

  static void BboxTransform(DeviceCtx* ctx, int64_t m, const T* bbox, const T* target_bbox,
                            T* deltas) {
    // m = n * h * w * a
    // shape: bbox == target_bbox == deltas (n, h, w, a * 4)
    for (int64_t i = 0; i < m; ++i) {
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
    // m = n * h * w * a
    // shape: bbox == deltas == bbox_pred (n, h, w, a * 4)
    for (int64_t i = 0; i < m; ++i) {
      float w = bbox[i * 4 + 2] - bbox[i * 4] + 1.0f;
      float h = bbox[i * 4 + 3] - bbox[i * 1 + 1] + 1.0f;
      float ctr_x = bbox[i * 4] + 0.5f * w;
      float ctr_y = bbox[i * 4 + 1] + 0.5f * h;

      float pred_ctr_x = deltas[i * 4] * w + ctr_x;
      float pred_ctr_y = deltas[i * 4 + 1] * h + ctr_y;
      float pred_w = std::exp(deltas[i * 4 + 2]) * w;
      float pred_h = std::exp(deltas[i * 4 + 3]) * h;

      bbox_pred[i * 4 + 0] = pred_ctr_x - 0.5f * pred_w;
      bbox_pred[i * 4 + 1] = pred_ctr_y - 0.5f * pred_h;
      bbox_pred[i * 4 + 2] = pred_ctr_x + 0.5f * pred_w;
      bbox_pred[i * 4 + 3] = pred_ctr_y + 0.5f * pred_h;
    }
  }

  static void ClipBoxes(DeviceCtx* ctx, int64_t n, int64_t m, const T* im_info, T* proposals) {
    // m = h * w * a
    // im_info (n, 3) ; 3 -> (origin_h, origin_w, scale)
    // proposals (n, h, w, a * 4)
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < m; ++j) {
        int64_t x1 = i * m * 4 + j;
        int64_t y1 = i * m * 4 + j + 1;
        int64_t x2 = i * m * 4 + j + 2;
        int64_t y2 = i * m * 4 + j + 3;
        T oh = im_info[n * 3] - 1;
        T ow = im_info[n * 3 + 1] - 1;
        // x1 >= 0 && y1 >= 0 && x2 <= origin_width && y2 <= origin_height
        proposals[x1] = std::max(std::min(proposals[x1], ow), static_cast<T>(0));
        proposals[y1] = std::max(std::min(proposals[y1], oh), static_cast<T>(0));
        proposals[x2] = std::max(std::min(proposals[x2], ow), static_cast<T>(0));
        proposals[y2] = std::max(std::min(proposals[y2], oh), static_cast<T>(0));
      }
    }
  }

  static std::vector<int64_t> FilterBoxesByMinSize(DeviceCtx* ctx, int64_t n, int64_t m,
                                                   int32_t min_size, const T* im_info,
                                                   T* proposals) {
    // m = h * w * a
    std::vector<int64_t> keep_to(n);
    for (int64_t i = 0; i < n; ++i) {
      T real_min_size = im_info[n * 3 + 2] * min_size;
      int64_t itor = 0;
      int64_t filter_from = m;
      while (itor != filter_from) {
        int64_t cur_x1 = i * m * 4 + itor;
        int64_t cur_y1 = i * m * 4 + itor + 1;
        int64_t cur_x2 = i * m * 4 + itor + 2;
        int64_t cur_y2 = i * m * 4 + itor + 3;
        if (proposals[cur_x2] - proposals[cur_x1] + 1 < real_min_size
            || proposals[cur_y2] - proposals[cur_y1] + 1 < real_min_size) {
          --filter_from;
          if (itor == filter_from) { break; }
          std::swap_ranges(proposals + cur_x1, proposals + cur_x1 + 4,
                           proposals + i * m * 4 + filter_from);
        } else {
          ++itor;
        }
      }
      keep_to[i] = filter_from;
    }
    return keep_to;
  }

  static void SortByScore(DeviceCtx* ctx, int64_t n, int64_t m, const std::vector<int64_t>& keep_to,
                          T* fg_scores, T* proposals) {
    // fg_scores (n, h * w * a, 1)
    // proposals (n, h * w * a, 4)
    // m = h * w * a
    for (int64_t i = 0; i < n; ++i) {
      int64_t keep = keep_to.at(i);
      T* cur_scores = fg_scores + i * m;
      T* cur_proposals = proposals + i * m * 4;
      std::vector<size_t> sort_indexes(keep);
      std::iota(sort_indexes.begin(), sort_indexes.end(), 0);
      sort(sort_indexes.begin(), sort_indexes.end(),
           [&](size_t idx1, size_t idx2) { return cur_scores[idx1] > cur_scores[idx2]; });
      for (int64_t j = 0; j < sort_indexes.size(); ++j) {
        int64_t idx = j;
        int64_t val = sort_indexes[j];
        while (sort_indexes[idx] != val) {
          std::iter_swap(cur_scores + idx, cur_scores + val);
          std::swap_ranges(cur_proposals + idx * 4, cur_proposals + (idx + 1) * 4,
                           cur_proposals + val * 4);
          val = idx;
          idx = std::find(sort_indexes.begin(), sort_indexes.end(), idx) - sort_indexes.begin();
        }
      }
    }
  }

  static void Nms(DeviceCtx* ctx, int64_t n, int64_t m, const std::vector<int64_t>& keep_to,
                  int64_t pre_nms_top_n, int64_t post_nms_top_n, float nms_threshold,
                  const T* proposals, const T* fg_scores, T* rois, T* scores) {
    // proposals (n, h * w * a, 4)
    // fg_scores (n, h * w * a, 1)
    // rois (n, post_nms_top_n, 4)
    // scores (n, post_nms_top_n, 1)
    // m = h * w * a
    for (int64_t i = 0; i < n; ++i) {
      int64_t topn = pre_nms_top_n == -1 ? keep_to.at(i) : std::min(keep_to.at(i), pre_nms_top_n);
      std::vector<bool> suppressed(topn, false);
      std::vector<int64_t> keep;
      for (int64_t j = 0; j < topn; ++j) {
        if (suppressed[j]) { continue; }
        keep.push_back(j);
        const T* cur_prop = proposals + (i * m + j) * 4;
        for (int64_t k = j + 1; k < topn; ++k) {
          const T* cand_prop = proposals + (i * m + k) * 4;
          // float iou = RcnnUtil<T>::InterOverUnion(cur_prop, cand_prop);
          float iou = (*cur_prop) + (*cand_prop);
          if (iou >= nms_threshold) { suppressed[k] = true; }
        }
      }
      for (int64_t l = 0; l < std::min(keep.size(), static_cast<size_t>(post_nms_top_n)); ++l) {
        for (int64_t h = 0; h < 4; ++h) {
          rois[(i * post_nms_top_n + l) * 4 + h] = proposals[(i * m + keep[l]) * 4 + h];
        }
        scores[i * post_nms_top_n + l] = fg_scores[i * m + keep[l]];
      }
    }
  }
};

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalConf, ProposalKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
