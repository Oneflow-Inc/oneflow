#include <opencv2/opencv.hpp>
#include "oneflow/core/kernel/image_segmentation_mask_kernel.h"
#include "oneflow/core/kernel/bbox_util.h"
#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

namespace {

template<typename T>
T* GetPointerPtr(T* ptr, int32_t width, int32_t height, int32_t x0, int32_t y0) {
  CHECK_LT(x0, width);
  CHECK_LT(y0, height);
  return ptr + y0 * width + x0;
}

template<typename T, typename U = T>
void CopyRegion(T* dst_ptr, size_t dst_step, const T* src_ptr, size_t src_step, size_t region_width,
                size_t region_height) {
  FOR_RANGE(int32_t, i, 0, region_height) {
    CopyElem(src_ptr + i * src_step, dst_ptr + i * dst_step, region_width);
  }
}

template<typename T>
void InitPaddedMask(const int32_t dim0_idx, const int32_t padding, const Blob* mask_blob,
                    Blob* padded_mask_blob) {
  const T* mask_ptr = mask_blob->dptr<T>(dim0_idx);
  T* padded_mask_ptr = padded_mask_blob->mut_dptr<T>(dim0_idx);
  int32_t mask_h = mask_blob->static_shape().At(1);
  int32_t mask_w = mask_blob->static_shape().At(2);
  int32_t pad_mask_h = padded_mask_blob->static_shape().At(1);
  int32_t pad_mask_w = padded_mask_blob->static_shape().At(2);
  CHECK_EQ(mask_h + padding * 2, pad_mask_h);
  CHECK_EQ(mask_w + padding * 2, pad_mask_w);
  CopyRegion(GetPointerPtr<T>(padded_mask_ptr, pad_mask_w, pad_mask_h, padding, padding),
             pad_mask_w, GetPointerPtr<const T>(mask_ptr, mask_w, mask_h, 0, 0), mask_w, mask_w,
             mask_h);
}

template<typename T>
void ExpandBox(const int32_t dim0_idx, const int32_t mask_h, const int32_t mask_w,
               const int32_t padding, const Blob* bbox_blob, BBoxT<int32_t>* expanded_bbox) {
  const float scale_h = (mask_h + 2.0 * padding) / mask_h;
  const float scale_w = (mask_w + 2.0 * padding) / mask_w;
  const auto* bbox = BBoxT<T>::Cast(bbox_blob->dptr<T>(dim0_idx));
  const T center_x = (bbox->right() + bbox->left()) * 0.5;
  const T center_y = (bbox->bottom() + bbox->top()) * 0.5;
  const T half_w = (bbox->right() - bbox->left()) * 0.5 * scale_w;
  const T half_h = (bbox->bottom() - bbox->top()) * 0.5 * scale_h;
  expanded_bbox->set_ltrb(center_x - half_w, center_y - half_h, center_x + half_w,
                          center_y + half_h);
}

template<typename T>
int GetCVType();

template<>
int GetCVType<float>() {
  return CV_32FC1;
}
template<>
int GetCVType<double>() {
  return CV_64FC1;
}

template<typename T>
void ResizeMask(const BBoxT<int32_t>* bbox, Blob* padded_mask_blob, cv::Mat* ret_mat) {
  const int32_t mask_h = padded_mask_blob->shape().At(1);
  const int32_t mask_w = padded_mask_blob->shape().At(2);
  int32_t bbox_h = bbox->height();
  int32_t bbox_w = bbox->width();
  bbox_h = std::max(bbox_h, 1);
  bbox_w = std::max(bbox_w, 1);

  cv::Mat origin_mat(mask_w, mask_h, GetCVType<T>(), padded_mask_blob->mut_dptr<T>());
  const auto& size = cv::Size(bbox_w, bbox_h);
  cv::Mat target_mat(size, GetCVType<T>());
  cv::resize(origin_mat, target_mat, size, 0, 0, cv::INTER_LINEAR);
  *ret_mat = target_mat;
}

void BinarizeMask(const cv::Mat& mat, const float threshold, std::vector<uint8_t>& binary_mask) {
  binary_mask.resize(mat.rows * mat.cols);
  CHECK(mat.isContinuous());
  CHECK(mat.type() == CV_32FC1);
  uint8_t* binary_mask_ptr = binary_mask.data();
  FOR_RANGE(int32_t, i, 0, mat.rows * mat.cols) {
    binary_mask_ptr[i] = mat.at<float>(i) > threshold;
  }
}

void CopyToOutputBlob(int32_t dim0_idx, const Blob* image_size_blob,
                      const std::vector<uint8_t>& binary_mask, const BBoxT<int32_t>* bbox,
                      Blob* out_blob) {
  uint8_t* im_mask_ptr = out_blob->mut_dptr<uint8_t>(dim0_idx);
  const int32_t im_idx = out_blob->record_id_in_device_piece(dim0_idx);
  const int32_t im_height = image_size_blob->dptr<int32_t>(im_idx)[0];
  const int32_t im_width = image_size_blob->dptr<int32_t>(im_idx)[1];
  const int32_t bbox_h = bbox->height();
  const int32_t bbox_w = bbox->width();
  CHECK_EQ(bbox_w * bbox_h, binary_mask.size());
  CHECK_LE(im_height * im_width, out_blob->static_shape().Count(1));

  const int32_t clipped_x0 = std::max<int32_t>(bbox->left(), 0);
  const int32_t clipped_y0 = std::max<int32_t>(bbox->top(), 0);
  const int32_t clipped_w = std::min<int32_t>(bbox->right() + 1, im_width) - clipped_x0;
  const int32_t clipped_h = std::min<int32_t>(bbox->bottom() + 1, im_height) - clipped_y0;
  const int32_t clipped_left = clipped_x0 - bbox->left();
  const int32_t clipped_top = clipped_y0 - bbox->top();

  CopyRegion(
      GetPointerPtr<uint8_t>(im_mask_ptr, im_width, im_height, clipped_x0, clipped_y0), im_width,
      GetPointerPtr<const uint8_t>(binary_mask.data(), bbox_w, bbox_h, clipped_left, clipped_top),
      bbox_w, clipped_w, clipped_h);
}

template<typename T>
void ClearBlob(Blob* blob) {
  std::memset(blob->mut_dptr<T>(), 0, blob->static_shape().elem_cnt() * sizeof(T));
}

}  // namespace

template<typename T>
void ImageSegmentationMaskKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().image_segmentation_mask_conf();
  const Blob* mask_blob = BnInOp2Blob("mask");
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* image_size_blob = BnInOp2Blob("image_size");
  Blob* padded_mask_blob = BnInOp2Blob("padded_mask");
  Blob* out_blob = BnInOp2Blob("out");

  ClearBlob<T>(padded_mask_blob);
  ClearBlob<uint8_t>(out_blob);

  MultiThreadLoop(out_blob->shape().At(0), [&](int64_t i) {
    cv::Mat mask_mat;
    std::array<int32_t, 4> expanded_bbox_vec;
    std::vector<uint8_t> binarized_mask;

    auto* expanded_bbox = BBoxT<int32_t>::Cast(expanded_bbox_vec.data());
    InitPaddedMask<T>(i, conf.padding(), mask_blob, padded_mask_blob);
    ExpandBox<T>(i, mask_blob->static_shape().At(1), mask_blob->static_shape().At(2),
                 conf.padding(), bbox_blob, expanded_bbox);
    ResizeMask<T>(expanded_bbox, padded_mask_blob, &mask_mat);
    BinarizeMask(mask_mat, conf.threshold(), binarized_mask);
    CopyToOutputBlob(i, image_size_blob, binarized_mask, expanded_bbox, out_blob);
  });
}

template<typename T>
void ImageSegmentationMaskKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->set_dim0_valid_num(0, BnInOp2Blob("mask")->dim0_valid_num(0));
}

template<typename T>
void ImageSegmentationMaskKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyRecordIdInDevicePieceFrom(ctx.device_ctx, BnInOp2Blob("mask"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kImageSegmentationMaskConf,
                               ImageSegmentationMaskKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
