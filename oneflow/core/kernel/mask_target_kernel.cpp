#include "oneflow/core/kernel/mask_target_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/record/coco.pb.h"
#include "oneflow/core/kernel/rle_util.h"

namespace oneflow {

namespace {

template<typename T>
void ScaleSegmPolygonLists(const Blob* scale_blob, std::vector<std::vector<PolygonList>>* segms) {
  FOR_RANGE(size_t, n, 0, segms->size()) {
    const T y_scale = scale_blob->dptr<T>(n)[0];
    const T x_scale = scale_blob->dptr<T>(n)[1];
    FOR_RANGE(size_t, g, 0, segms->at(n).size()) {
      FOR_RANGE(size_t, p, 0, segms->at(n).at(g).polygons_size()) {
        FOR_RANGE(size_t, i, 0, segms->at(n).at(g).polygons(p).value_size()) {
          const T scale = i % 2 == 0 ? x_scale : y_scale;
          segms->at(n).at(g).mutable_polygons(p)->set_value(
              i, scale * segms->at(n).at(g).polygons(p).value(i));
        }
      }
    }
  }
}

}  // namespace

template<typename T>
void MaskTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  static_assert(SegmBBox::ElemCnt == 4, "number of elements in SegmBBox should be 4");
  static_assert(RoiBBox::ElemCnt == 5, "number of elements in RoiBBox should be 5");
  std::vector<std::vector<PolygonList>> segms;
  // input blobs
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* labels_blob = BnInOp2Blob("labels");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const Blob* gt_segms_blob = BnInOp2Blob("gt_segm_polygon_lists");
  const Blob* im_scale_blob = BnInOp2Blob("im_scale");
  // output blobs
  Blob* masks_blob = BnInOp2Blob("masks");
  Blob* mask_rois_blob = BnInOp2Blob("mask_rois");
  Blob* mask_labels_blob = BnInOp2Blob("mask_labels");
  // const buf blobs
  const auto mask_h = static_cast<size_t>(masks_blob->static_shape().At(1));
  const auto mask_w = static_cast<size_t>(masks_blob->static_shape().At(2));
  const bool input_blob_has_record_id = rois_blob->has_record_id_in_device_piece_field();
  CHECK_EQ(labels_blob->has_record_id_in_device_piece_field(), input_blob_has_record_id);

  ParseSegmPolygonLists(gt_segms_blob, &segms);
  ScaleSegmPolygonLists<T>(im_scale_blob, &segms);

  const int64_t num_rois = rois_blob->shape().At(0);
  CHECK_GT(num_rois, 0);
  CHECK_EQ(labels_blob->shape().At(0), num_rois);
  int64_t mask_idx = 0;
  // set dim0_valid_num to max
  masks_blob->set_dim0_valid_num(0, num_rois);
  mask_rois_blob->set_dim0_valid_num(0, num_rois);
  const RoiBBox* roi_bboxes = RoiBBox::Cast(rois_blob->dptr<T>());
  RoiBBox* mask_roi_bboxes = RoiBBox::Cast(mask_rois_blob->mut_dptr<T>());
  const auto CopyRecordIdIfNeed = [&](const int64_t roi_idx, const int64_t mask_idx) {
    if (input_blob_has_record_id) {
      const int64_t record_id = rois_blob->record_id_in_device_piece(roi_idx);
      masks_blob->set_record_id_in_device_piece(mask_idx, record_id);
      mask_rois_blob->set_record_id_in_device_piece(mask_idx, record_id);
      mask_labels_blob->set_record_id_in_device_piece(mask_idx, record_id);
    }
  };
  FOR_RANGE(int64_t, roi_idx, 0, num_rois) {
    const int32_t label = *labels_blob->dptr<int32_t>(roi_idx);
    CHECK_GE(label, 0);
    if (label == 0) { continue; }
    const RoiBBox& fg_roi = roi_bboxes[roi_idx];
    const int32_t img_idx = fg_roi.index();
    CHECK_GE(img_idx, 0);
    CHECK_LT(img_idx, gt_boxes_blob->shape().At(0));
    CHECK_GE(gt_boxes_blob->dim1_valid_num(img_idx), 1);
    mask_roi_bboxes[mask_idx].elem() = fg_roi.elem();
    *mask_labels_blob->mut_dptr<int32_t>(mask_idx) = label;
    CopyRecordIdIfNeed(roi_idx, mask_idx);
    mask_idx += 1;
  }

  MultiThreadLoop(mask_idx, [&](const int64_t idx) {
    const RoiBBox& roi = mask_roi_bboxes[idx];
    const int32_t img_idx = roi.index();
    const size_t max_iou_gt_idx =
        GetMaxOverlapIndex(roi, SegmBBox::Cast(gt_boxes_blob->dptr<float>(img_idx)),
                           static_cast<size_t>(gt_boxes_blob->dim1_valid_num(img_idx)));
    Segm2Mask(segms.at(static_cast<size_t>(img_idx)).at(max_iou_gt_idx), roi, mask_h, mask_w,
              masks_blob->mut_dptr<int32_t>(idx));
  });

  masks_blob->set_dim0_valid_num(0, mask_idx);
  mask_rois_blob->set_dim0_valid_num(0, mask_idx);
  mask_labels_blob->set_dim0_valid_num(0, mask_idx);
}

template<typename T>
void MaskTargetKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing here because it will be set in ForwardDataContent
}

template<typename T>
void MaskTargetKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing here because it will be set in ForwardDataContent
}

template<typename T>
void MaskTargetKernel<T>::ParseSegmPolygonLists(
    const Blob* gt_segms, std::vector<std::vector<PolygonList>>* segms) const {
  segms->resize(static_cast<size_t>(gt_segms->shape().At(0)));
  FOR_RANGE(size_t, n, 0, gt_segms->shape().At(0)) {
    segms->at(n).resize(static_cast<size_t>(gt_segms->dim1_valid_num(n)));
    FOR_RANGE(size_t, g, 0, gt_segms->dim1_valid_num(n)) {
      CHECK(segms->at(n).at(g).ParseFromArray(gt_segms->dptr<char>(n, g),
                                              gt_segms->dim2_valid_num(n, g)));
    }
  }
}

template<typename T>
void MaskTargetKernel<T>::Segm2BBox(const PolygonList& segm, SegmBBox* bbox) const {
  CHECK_GE(segm.polygons_size(), 1);
  Polygon2BBox(segm.polygons(0), bbox);
  FOR_RANGE(int32_t, i, 1, segm.polygons_size()) {
    float one_bbox_elem[SegmBBox::ElemCnt];
    SegmBBox* one_bbox = SegmBBox::Cast(one_bbox_elem);
    Polygon2BBox(segm.polygons(i), one_bbox);
    bbox->set_ltrb(std::min(bbox->left(), one_bbox->left()), std::min(bbox->top(), one_bbox->top()),
                   std::max(bbox->right(), one_bbox->right()),
                   std::max(bbox->bottom(), one_bbox->bottom()));
  }
}

template<typename T>
void MaskTargetKernel<T>::Polygon2BBox(const FloatList& polygon, SegmBBox* bbox) const {
  const PbRf<float>& xy = polygon.value();
  CHECK_EQ(xy.size() % 2, 0);
  CHECK_GE(xy.size() / 2, 3);
  bbox->set_ltrb(xy[0], xy[1], xy[0], xy[1]);
  FOR_RANGE(int32_t, i, 1, polygon.value_size() / 2) {
    bbox->set_ltrb(std::min(bbox->left(), xy[i * 2]), std::min(bbox->top(), xy[i * 2 + 1]),
                   std::max(bbox->right(), xy[i * 2]), std::max(bbox->bottom(), xy[i * 2 + 1]));
  }
}

template<typename T>
void MaskTargetKernel<T>::Segm2Mask(const PolygonList& segm, const RoiBBox& fg_roi, size_t mask_h,
                                    size_t mask_w, int32_t* mask) const {
  CHECK_GE(segm.polygons_size(), 1);
  const size_t mask_elem_num = mask_h * mask_w;
  std::vector<uint8_t> mask_vec(mask_elem_num);
  std::vector<uint8_t> one_mask_vec;
  std::vector<double> xy;
  const double scale_w =
      static_cast<double>(mask_w) / std::max(OneVal<T>::value, fg_roi.right() - fg_roi.left());
  const double scale_h =
      static_cast<double>(mask_h) / std::max(OneVal<T>::value, fg_roi.bottom() - fg_roi.top());
  FOR_RANGE(int32_t, i, 0, segm.polygons_size()) {
    const PbRf<float>& polygon_xy = segm.polygons(i).value();
    CHECK_EQ(polygon_xy.size() % 2, 0);
    xy.resize(static_cast<size_t>(polygon_xy.size()));
    FOR_RANGE(int32_t, j, 0, polygon_xy.size()) {
      if (j % 2 == 0) {
        xy[j] = (polygon_xy[j] - fg_roi.left()) * scale_w;
      } else {
        xy[j] = (polygon_xy[j] - fg_roi.top()) * scale_h;
      }
    }

    if (i == 0) {
      RleUtil::PolygonXy2ColMajorMask(xy.data(), xy.size(), mask_h, mask_w, mask_vec.data());
    } else {
      one_mask_vec.resize(mask_elem_num);
      RleUtil::PolygonXy2ColMajorMask(xy.data(), xy.size(), mask_h, mask_w, one_mask_vec.data());
      std::transform(mask_vec.cbegin(), mask_vec.cend(), one_mask_vec.cbegin(), mask_vec.begin(),
                     std::bit_or<uint8_t>());
    }
  }

  FOR_RANGE(int32_t, r, 0, mask_h) {
    FOR_RANGE(int32_t, c, 0, mask_w) { mask[r * mask_w + c] = mask_vec[c * mask_h + r] ? 1 : 0; }
  }
}

template<typename T>
size_t MaskTargetKernel<T>::GetMaxOverlapIndex(const RoiBBox& fg_roi, const SegmBBox* gt_bboxs,
                                               size_t gt_bboxs_num) const {
  CHECK_GE(gt_bboxs_num, 1);
  size_t max_overlap_idx = 0;
  float max_overlap = fg_roi.InterOverUnion(&gt_bboxs[0]);
  FOR_RANGE(size_t, i, 1, gt_bboxs_num) {
    float overlap = fg_roi.InterOverUnion(&gt_bboxs[i]);
    if (overlap > max_overlap) {
      max_overlap_idx = i;
      max_overlap = overlap;
    }
  }
  return max_overlap_idx;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaskTargetConf, MaskTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
