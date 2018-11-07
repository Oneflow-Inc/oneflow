#include "oneflow/core/kernel/generate_anchors_kernel.h"
#include <cfenv>

namespace oneflow {

namespace {

// ratio = h / w
std::pair<float, float> CalcBoxHeightAndWidth(const GenerateAnchorsOpConf::Algorithm algorithm,
                                              const float scale, const float ratio,
                                              const float step) {
  float h = 0.0f;
  float w = 0.0f;
  switch (algorithm) {
    case GenerateAnchorsOpConf::kFloating: {
      w = scale / std::sqrt(ratio);
      h = scale * std::sqrt(ratio);
      break;
    }
    case GenerateAnchorsOpConf::kRoundRatioFirst: {
      int save_round_way = std::fegetround();
      CHECK_EQ(std::fesetround(FE_TONEAREST), 0);
      w = std::nearbyint(std::sqrt(step * step / ratio));
      h = std::nearbyint(w * ratio);
      const float rate = std::floor(scale / step);
      w *= rate;
      h *= rate;
      std::fesetround(save_round_way);
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  return std::make_pair(h, w);
}

template<typename It>
void SetBoxSize(It it, const GenerateAnchorsOpConf::Coordinate coordinate, const float step,
                const float height, const float width, const float img_height,
                const float img_width) {
  switch (coordinate) {
    case GenerateAnchorsOpConf::kNormXYXY: {
      const float base_ctr = 0.5f * step;
      *(it + 0) = (base_ctr - 0.5f * width) / img_width;
      *(it + 1) = (base_ctr - 0.5f * height) / img_height;
      *(it + 2) = (base_ctr + 0.5f * width) / img_width;
      *(it + 3) = (base_ctr + 0.5f * height) / img_height;
      break;
    }
    case GenerateAnchorsOpConf::kIntXYXY: {
      const float base_ctr = 0.5f * (step - 1);
      *(it + 0) = base_ctr - 0.5f * (width - 1);
      *(it + 1) = base_ctr - 0.5f * (height - 1);
      *(it + 2) = base_ctr + 0.5f * (width - 1);
      *(it + 3) = base_ctr + 0.5f * (height - 1);
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
}

float GetShiftStepDistance(const GenerateAnchorsOpConf::Coordinate coordinate, const float step,
                           const float norm) {
  switch (coordinate) {
    case GenerateAnchorsOpConf::kNormXYXY: {
      return step;
    }
    case GenerateAnchorsOpConf::kIntXYXY: {
      return step / norm;
    }
    default: { UNIMPLEMENTED(); }
  }
  return 0.0f;
}

template<typename T>
void ClipBox(T* anchor_box_ptr, const GenerateAnchorsOpConf::Coordinate coordinate,
             const int32_t img_h, const int32_t img_w) {
  switch (coordinate) {
    case GenerateAnchorsOpConf::kNormXYXY: {
      FOR_RANGE(size_t, i, 0, GenerateAnchorsKernel<T>::kBoxElemSize) {
        anchor_box_ptr[i] = std::min<T>(std::max<T>(anchor_box_ptr[i], 0.0f), 1.0f);
      }
      break;
    }
    case GenerateAnchorsOpConf::kIntXYXY: {
      anchor_box_ptr[0] = std::max<T>(std::min<T>(anchor_box_ptr[0], img_w - 1), 0);
      anchor_box_ptr[1] = std::max<T>(std::min<T>(anchor_box_ptr[1], img_h - 1), 0);
      anchor_box_ptr[2] = std::max<T>(std::min<T>(anchor_box_ptr[2], img_w - 1), 0);
      anchor_box_ptr[3] = std::max<T>(std::min<T>(anchor_box_ptr[3], img_h - 1), 0);
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
}

}  // namespace

template<typename T>
std::vector<T> GenerateAnchorsKernel<T>::GenerateBaseAnchors() const {
  std::vector<std::pair<float, float>> scale_ratio_pair_vec;
  const GenerateAnchorsOpConf& conf = op_conf().generate_anchors_conf();
  const float feat_step = conf.feature_map_stride();
  for (const auto& scale_conf : conf.scale_conf()) {
    for (float ratio : scale_conf.ratio()) {
      scale_ratio_pair_vec.emplace_back(std::make_pair(scale_conf.scale(), ratio));
    }
  }
  for (const auto& aspect_ratio_conf : conf.aspect_ratio_conf()) {
    for (float scale : aspect_ratio_conf.scale()) {
      scale_ratio_pair_vec.emplace_back(std::make_pair(scale, aspect_ratio_conf.ratio()));
    }
  }

  std::vector<T> base_anchors_vec(scale_ratio_pair_vec.size() * kBoxElemSize);
  auto it = base_anchors_vec.begin();
  for (const auto& pair : scale_ratio_pair_vec) {
    CHECK_GE(std::distance(it, base_anchors_vec.end()), kBoxElemSize);
    float h = 0.0f;
    float w = 0.0f;
    std::tie(h, w) = CalcBoxHeightAndWidth(conf.algorithm(), pair.first, pair.second, feat_step);
    SetBoxSize(it, conf.coordinate(), feat_step, h, w, conf.image_height(), conf.image_width());
    it += kBoxElemSize;
  }
  return base_anchors_vec;
}

template<typename T>
void GenerateAnchorsKernel<T>::ShiftAnchors(const std::vector<T>& base_anchors_vec,
                                            Blob* anchors_blob) const {
  const GenerateAnchorsOpConf& conf = op_conf().generate_anchors_conf();
  const float shift_w =
      GetShiftStepDistance(conf.coordinate(), conf.feature_map_stride(), conf.image_width());
  const float shift_h =
      GetShiftStepDistance(conf.coordinate(), conf.feature_map_stride(), conf.image_height());
  const int32_t feat_h = anchors_blob->shape().At(0);
  const int32_t feat_w = anchors_blob->shape().At(1);
  const int32_t num_anchors = anchors_blob->shape().At(2);
  CHECK_EQ(kBoxElemSize, anchors_blob->shape().At(3));
  T* anchors_ptr = anchors_blob->mut_dptr<T>();
  FOR_RANGE(int32_t, h, 0, feat_h) {
    FOR_RANGE(int32_t, w, 0, feat_w) {
      T* cur_feat_anchors_ptr = anchors_ptr + (h * feat_w + w) * num_anchors * kBoxElemSize;
      FOR_RANGE(int32_t, i, 0, num_anchors) {
        const int32_t index = i * kBoxElemSize;
        cur_feat_anchors_ptr[index + 0] = base_anchors_vec[index + 0] + w * shift_w;
        cur_feat_anchors_ptr[index + 1] = base_anchors_vec[index + 1] + h * shift_h;
        cur_feat_anchors_ptr[index + 2] = base_anchors_vec[index + 2] + w * shift_w;
        cur_feat_anchors_ptr[index + 3] = base_anchors_vec[index + 3] + h * shift_h;
        if (conf.clip()) {
          ClipBox(cur_feat_anchors_ptr + index, conf.coordinate(), conf.image_height(),
                  conf.image_width());
        }
      }
    }
  }
}

template<typename T>
void GenerateAnchorsKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* anchors_blob = BnInOp2Blob("anchors");
  auto base_anchors_vec = GenerateBaseAnchors();
  CHECK_EQ(base_anchors_vec.size(), anchors_blob->shape().Count(2));
  ShiftAnchors(base_anchors_vec, anchors_blob);
}

template<typename T>
void GenerateAnchorsKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

/*
template<typename T>
void GenerateAnchorsKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK_EQ(BBox::ElemCnt, 4);
  const float fm_stride = conf.feature_map_stride();
  const int32_t height = std::ceil(conf.image_height() / fm_stride);
  const int32_t width = std::ceil(conf.image_width() / fm_stride);
  const int32_t scales_size = conf.anchor_scales_size();
  const int32_t ratios_size = conf.aspect_ratios_size();
  const int32_t num_anchors = scales_size * ratios_size;

  const float base_ctr = 0.5 * (fm_stride - 1);
  std::vector<T> base_anchors_vec(num_anchors * BBox::ElemCnt);
  // scale first, ratio last
  std::fesetround(FE_TONEAREST);
  FOR_RANGE(int32_t, i, 0, ratios_size) {
    const int32_t wr = std::nearbyint(std::sqrt(fm_stride * fm_stride / conf.aspect_ratios(i)));
    const int32_t hr = std::nearbyint(wr * conf.aspect_ratios(i));
    FOR_RANGE(int32_t, j, 0, scales_size) {
      const float scale = conf.anchor_scales(j) / fm_stride;
      const int32_t ws = wr * scale;
      const int32_t hs = hr * scale;
      auto* base_anchor_bbox = BBox::Cast(base_anchors_vec.data()) + i * scales_size + j;
      base_anchor_bbox->set_ltrb(base_ctr - 0.5 * (ws - 1), base_ctr - 0.5 * (hs - 1),
                                 base_ctr + 0.5 * (ws - 1), base_ctr + 0.5 * (hs - 1));
    }
  }

  const auto* base_anchors = BBox::Cast(base_anchors_vec.data());
  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      auto* anchor_bbox = BBox::Cast(anchors_ptr) + (h * width + w) * num_anchors;
      FOR_RANGE(int32_t, i, 0, num_anchors) {
        anchor_bbox[i].set_ltrb(
            base_anchors[i].left() + w * fm_stride, base_anchors[i].top() + h * fm_stride,
            base_anchors[i].right() + w * fm_stride, base_anchors[i].bottom() + h * fm_stride);
      }
    }
  }
  return num_anchors * height * width;
}



template<typename T>
void GenerateAnchorsKernel<T>::GenerateBoxes(const PriorBoxOpConf& conf, const int32_t img_num,
                                      const int32_t height, const int32_t width,
                                      Blob* boxes_blob) const {
  const float min_size = conf.min_size();
  const float max_size = conf.max_size();
  const int img_width = conf.img_width();
  const int img_height = conf.img_height();
  const int32_t ratios_size = conf.aspect_ratios_size();
  const int32_t num_anchors = conf.flip() ? (ratios_size * 2 + 2) : (ratios_size + 2);
  const bool use_clip = conf.clip();
  float step_w = img_width / width;
  float step_h = img_height / height;
  if (conf.has_step()) {
    step_w = conf.step();
    step_h = conf.step();
  }
  std::vector<float> box_ratios;
  box_ratios.push_back(1.0);
  box_ratios.push_back(-1.0);
  FOR_RANGE(int32_t, i, 0, ratios_size) {
    box_ratios.push_back(conf.aspect_ratios(i));
    if (conf.flip()) { box_ratios.push_back(1.0 / conf.aspect_ratios(i)); }
  }
  const float base_ctr_x = 0.5 * step_w;
  const float base_ctr_y = 0.5 * step_h;
  std::vector<T> base_anchors(num_anchors * 4);
  BBox<T>* base_anchor_bbox = BBox<T>::MutCast(base_anchors.data());
  // anchors order: 1.0、min_size*max_size 、ratios
  FOR_RANGE(int32_t, i, 0, num_anchors) {
    float box_width = 0;
    float box_height = 0;
    if (i == 1) {
      box_width = sqrt(min_size * max_size);
      box_height = box_width;
    } else {
      box_width = min_size * sqrt(box_ratios[i]);
      box_height = min_size / sqrt(box_ratios[i]);
    }
    BBox<T>* cur_anchor_bbox = base_anchor_bbox + i;
    cur_anchor_bbox->set_x1(Clip((base_ctr_x - 0.5 * box_width) / img_width, use_clip));
    cur_anchor_bbox->set_y1(Clip((base_ctr_y - 0.5 * box_height) / img_height, use_clip));
    cur_anchor_bbox->set_x2(Clip((base_ctr_x + 0.5 * box_width) / img_width, use_clip));
    cur_anchor_bbox->set_y2(Clip((base_ctr_y + 0.5 * box_height) / img_height, use_clip));
  }
  const BBox<T>* const_base_anchor_bbox = BBox<T>::Cast(base_anchors.data());
  FOR_RANGE(int32_t, n, 0, img_num) {
    FOR_RANGE(int32_t, h, 0, height) {
      FOR_RANGE(int32_t, w, 0, width) {
        FOR_RANGE(int32_t, i, 0, num_anchors) {
          BBox<T>* anchor_bbox = BBox<T>::MutCast(
              boxes_blob->mut_dptr<T>(n, h * width * num_anchors + w * num_anchors + i));
          anchor_bbox->set_x1(const_base_anchor_bbox[i].x1() + w * step_w / img_width);
          anchor_bbox->set_y1(const_base_anchor_bbox[i].y1() + h * step_h / img_height);
          anchor_bbox->set_x2(const_base_anchor_bbox[i].x2() + w * step_w / img_width);
          anchor_bbox->set_y2(const_base_anchor_bbox[i].y2() + h * step_h / img_height);
        }
      }
    }
  }
  LOG(INFO) << "for gdb";
}

template<typename T>
T GenerateAnchorsKernel<T>::Clip(const T value, const bool use_clip) const {
  TODO();
}
*/

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kGenerateAnchorsConf, GenerateAnchorsKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow