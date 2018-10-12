#include <cfenv>
#include "oneflow/core/kernel/bbox_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

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
    post_nms_bbox_inds.mut_index()[keep_num++] = pre_nms_bbox_inds.GetIndex(pre_nms_i);
    if (keep_num == post_nms_bbox_inds.size()) { break; }
  }
  post_nms_bbox_inds.Truncate(keep_num);

  CHECK_LE(post_nms_bbox_inds.size(), pre_nms_bbox_inds.size());
}

}  // namespace oneflow
