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
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(size_t, i, 0, out_blob->shape().At(0)) {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx, out_blob->mut_dptr<T>(i), anchors_blob->dptr<T>(),
                             anchors_blob->shape().elem_cnt() * sizeof(T));
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kGenerateAnchorsConf, GenerateAnchorsKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow