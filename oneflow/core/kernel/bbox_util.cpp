#include <cfenv>
#include "oneflow/core/kernel/bbox_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename BBox>
size_t BBoxUtil<BBox>::GenerateAnchors(const AnchorGeneratorConf& conf, T* anchors_ptr) {
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
      auto* base_anchor_bbox = BBox::MutCast(base_anchors_vec.data()) + i * scales_size + j;
      base_anchor_bbox->set_corner_coord(base_ctr - 0.5 * (ws - 1), base_ctr - 0.5 * (hs - 1),
                                         base_ctr + 0.5 * (ws - 1), base_ctr + 0.5 * (hs - 1));
    }
  }

  const auto* base_anchors = BBox::Cast(base_anchors_vec.data());
  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      auto* anchor_bbox = BBox::MutCast(anchors_ptr) + (h * width + w) * num_anchors;
      FOR_RANGE(int32_t, i, 0, num_anchors) {
        anchor_bbox[i].set_corner_coord(
            base_anchors[i].left() + w * fm_stride, base_anchors[i].top() + h * fm_stride,
            base_anchors[i].right() + w * fm_stride, base_anchors[i].bottom() + h * fm_stride);
      }
    }
  }
  return num_anchors * height * width;
}

template<typename BBox>
void BBoxUtil<BBox>::Nms(float thresh, const BBoxIndicesT& pre_nms_bbox_inds,
                         BBoxIndicesT& post_nms_bbox_inds) {
  CHECK_NE(pre_nms_bbox_inds.index(), post_nms_bbox_inds.index());
  CHECK_EQ(pre_nms_bbox_inds.bbox(), post_nms_bbox_inds.bbox());

  size_t keep_num = 0;
  auto IsSuppressed = [&](size_t pre_nms_n) -> bool {
    const auto* cur_bbox = pre_nms_bbox_inds.GetBBox(pre_nms_n);
    FOR_RANGE(size_t, post_nms_i, 0, keep_num) {
      const auto* keep_bbox = post_nms_bbox_inds.GetBBox(post_nms_i);
      if (keep_bbox->InterOverUnion(cur_bbox) >= thresh) { return true; }
    }
    return false;
  };
  FOR_RANGE(size_t, pre_nms_i, 0, pre_nms_bbox_inds.size()) {
    if (IsSuppressed(pre_nms_i)) { continue; }
    post_nms_bbox_inds.index()[keep_num++] = pre_nms_bbox_inds.GetIndex(pre_nms_i);
    if (keep_num == post_nms_bbox_inds.size()) { break; }
  }
  post_nms_bbox_inds.Truncate(keep_num);

  CHECK_LE(post_nms_bbox_inds.size(), pre_nms_bbox_inds.size());
}

#define INITIATE_BBOX_UTIL(T, type_cpp)                                         \
  template struct BBoxUtil<BBoxImpl<T, BBoxBase, BBoxCoord::kCorner>>;          \
  template struct BBoxUtil<BBoxImpl<T, ImIndexedBBoxBase, BBoxCoord::kCorner>>; \
  template struct BBoxUtil<BBoxImpl<const T, BBoxBase, BBoxCoord::kCorner>>;    \
  template struct BBoxUtil<BBoxImpl<const T, ImIndexedBBoxBase, BBoxCoord::kCorner>>;

OF_PP_FOR_EACH_TUPLE(INITIATE_BBOX_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
