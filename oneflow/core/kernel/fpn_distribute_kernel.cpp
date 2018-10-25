#include "oneflow/core/kernel/fpn_distribute_kernel.h"
//#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

namespace {

template<typename T>
int32_t BBoxFpnLevel(const IndexedBBoxT<T>* box, const int32_t roi_min_level,
                     const int32_t roi_max_level, const T roi_canonical_level,
                     const T roi_canonical_scale) {
  int32_t target_level =
      std::floor(roi_canonical_level
                 + std::log(std::sqrt(box->Area()) / roi_canonical_scale + 1e-6) / std::log(2));
  target_level = std::min(roi_max_level, target_level);
  target_level = std::max(roi_min_level, target_level);
  return static_cast<int32_t>(target_level);
}

}  // namespace

template<typename T>
void FpnDistributeKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const FpnDistributeOpConf& conf = this->op_conf().fpn_distribute_conf();
  const Blob* collected_rois_blob = BnInOp2Blob("collected_rois");
  int32_t level_count = conf.roi_max_level() - conf.roi_min_level() + 1;
  std::vector<Blob*> rois_blob_vec(level_count);
  FOR_RANGE(int32_t, idx, 0, level_count) {
    rois_blob_vec[idx] = BnInOp2Blob("rois_" + std::to_string(idx));
  }
  Blob* roi_indices_blob = BnInOp2Blob("roi_indices");
  roi_indices_blob->set_dim0_valid_num(0, collected_rois_blob->shape().At(0));
  Blob* roi_indices_buf_blob = BnInOp2Blob("roi_indices_buf");
  int32_t roi_indices_size = roi_indices_blob->shape().At(0);
  IndexSequence roi_indices(roi_indices_size, roi_indices_size,
                            roi_indices_blob->mut_dptr<int32_t>(), false);
  IndexSequence roi_indices_buf(roi_indices_size, roi_indices_size,
                                roi_indices_buf_blob->mut_dptr<int32_t>(), true);
  Blob* target_levels_blob = BnInOp2Blob("target_levels");
  std::vector<int32_t> level_copy_idx(level_count, 0);
  size_t roi_size = 5 * sizeof(T);
  FOR_RANGE(int64_t, collected_roi_idx, 0, collected_rois_blob->shape().At(0)) {
    const T* collected_roi_ptr = collected_rois_blob->dptr<T>(collected_roi_idx);
    const auto* roi_bbox = IndexedBBoxT<T>::Cast(collected_roi_ptr);
    int32_t target_level = BBoxFpnLevel<T>(roi_bbox, conf.roi_min_level(), conf.roi_max_level(),
                                           static_cast<T>(conf.roi_canonical_level()),
                                           static_cast<T>(conf.roi_canonical_scale()));
    target_levels_blob->mut_dptr<int32_t>()[collected_roi_idx] = target_level;
    int32_t target_level_offset = target_level - conf.roi_min_level();
    Memcpy<DeviceType::kCPU>(
        ctx.device_ctx,
        rois_blob_vec[target_level_offset]->mut_dptr<T>() + level_copy_idx[target_level_offset],
        collected_roi_ptr, roi_size);
    rois_blob_vec[target_level_offset]->set_record_id_in_device_piece(
        level_copy_idx[target_level_offset] / 5,
        collected_rois_blob->record_id_in_device_piece(collected_roi_idx));
    level_copy_idx[target_level_offset] += 5;
  }
  int32_t roi_indices_idx = 0;
  FOR_RANGE(int64_t, target_level_idx, 0, level_count) {
    int32_t target_level = target_level_idx + conf.roi_min_level();
    FOR_RANGE(int64_t, collected_roi_idx, 0, roi_indices_blob->shape().At(0)) {
      if (target_level == target_levels_blob->dptr<int32_t>()[collected_roi_idx]) {
        roi_indices_blob->mut_dptr<int32_t>()[roi_indices_idx++] = collected_roi_idx;
      }
    }
  }
  roi_indices.ArgSort(roi_indices_buf);
  FOR_RANGE(int64_t, target_level_idx, 0, level_count) {
    rois_blob_vec[target_level_idx]->set_dim0_valid_num(0, level_copy_idx[target_level_idx] / 5);
  }
}

template<typename T>
void FpnDistributeKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing here because it will be set in ForwardDataContent
}

template<typename T>
void FpnDistributeKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // record indices of rois will be set in ForwardDataContent
  BnInOp2Blob("roi_indices")
      ->CopyRecordIdInDevicePieceFrom(ctx.device_ctx, BnInOp2Blob("collected_rois"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnDistributeConf, FpnDistributeKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
