#include <opencv2/opencv.hpp>
#include "oneflow/core/kernel/rle_segmentation_result_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/rle_util.h"

namespace oneflow {

namespace {

template<typename T>
T* GetPointerPtr(T* ptr, int32_t width, int32_t height, int32_t x0, int32_t y0) {
  TODO();
  return nullptr;
}

template<typename T>
void CopyRegion(T* dst_ptr, size_t dst_step, const T* src_ptr, size_t src_step, size_t region_width,
                size_t region_height) {
  TODO();
}

template<typename T>
void InitPaddedMask(Blob* padded_mask_blob, const Blob* mask_blob, const Blob* roi_labels_blob,
                    int32_t dim0_idx) {
  TODO();
}

void ExpandRoi(BBox<float>* expanded_bbox, const Blob* rois_blob, int32_t dim0_idx,
               const Shape& mask_shape) {
  TODO();
}

void Resize(cv::Mat* img, Blob* padded_mask_blob, const BBox<float>* bbox) {
  cv::Mat mask_img(padded_mask_blob->shape().At(1), padded_mask_blob->shape().At(0), CV_8UC1,
                   padded_mask_blob->mut_dptr<uint8_t>());
  cv::resize(mask_img, *img, cv::Size(bbox->width(), bbox->height()), 0, 0, cv::INTER_LINEAR);
}

void CopyToImMask(Blob* im_mask_blob, const cv::Mat& img, const BBox<float>* expanded_bbox) {
  CHECK(img.isContinuous());
  uint8_t* im_mask_ptr = im_mask_blob->mut_dptr<uint8_t>();
  size_t width = im_mask_blob->shape().At(1);
  size_t height = im_mask_blob->shape().At(0);
  CHECK_LE(img.rows, height);
  CHECK_LE(img.cols, width);
  int32_t x0 = std::max<int32_t>(expanded_bbox->x1(), 0);
  int32_t y0 = std::max<int32_t>(expanded_bbox->y1(), 0);
  int32_t w = std::min<int32_t>(expanded_bbox->x2() + 1, width) - x0;
  int32_t h = std::min<int32_t>(expanded_bbox->y2() + 1, height) - y0;
  int32_t im_x0 = x0 - expanded_bbox->x1();
  int32_t im_y0 = y0 - expanded_bbox->y1();
  CopyRegion(GetPointerPtr<uint8_t>(im_mask_ptr, width, height, x0, y0), width,
             GetPointerPtr<const uint8_t>(img.data, img.cols, img.rows, im_x0, im_y0), img.cols, w,
             h);
}

void RleEncodeIntoOutputblob(Blob* out_blob, const Blob* im_mask_blob, int32_t dim0_idx) {
  size_t height = im_mask_blob->shape().At(0);
  size_t width = im_mask_blob->shape().At(1);
  size_t len = RleEncode(out_blob->mut_dptr<uint32_t>(dim0_idx), im_mask_blob->dptr<uint8_t>(),
                         height, width);
  CHECK_EQ(len, out_blob->shape().Count(1));
}

}  // namespace

template<typename T>
void RleSegmentationResultKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* mask_blob = BnInOp2Blob("masks");
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* roi_labels_blob = BnInOp2Blob("roi_labels");
  Blob* padded_mask_blob = BnInOp2Blob("padded_mask");
  Blob* im_mask_blob = BnInOp2Blob("im_mask");
  Blob* out_blob = BnInOp2Blob("out");
  FOR_RANGE(int32_t, i, 0, mask_blob->shape().At(0)) {
    cv::Mat img;
    std::array<float, 4> expanded_bbox;
    auto* expanded_roi = BBox<float>::MutCast(&expanded_bbox[0]);

    InitPaddedMask<T>(padded_mask_blob, mask_blob, roi_labels_blob, i);
    ExpandRoi(expanded_roi, rois_blob, i, mask_blob->shape());
    Resize(&img, padded_mask_blob, expanded_roi);
    CopyToImMask(im_mask_blob, img, expanded_roi);
    RleEncodeIntoOutputblob(out_blob, im_mask_blob, i);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kRleSegmentationResultConf,
                               RleSegmentationResultKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
