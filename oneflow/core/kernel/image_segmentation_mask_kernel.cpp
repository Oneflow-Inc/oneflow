#include <opencv2/opencv.hpp>
#include "oneflow/core/kernel/image_segmentation_mask_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
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
void InitPaddedMask(Blob* padded_mask_blob, const Blob* mask_blob, const Blob* roi_labels_blob,
                    int32_t dim0_idx) {
  int32_t class_id = *roi_labels_blob->dptr<int32_t>(dim0_idx);
  CHECK_GT(class_id, 0);
  const T* mask_ptr = mask_blob->dptr<T>(dim0_idx, class_id);
  T* padded_mask_ptr = padded_mask_blob->mut_dptr<T>();
  int32_t width = mask_blob->shape().At(3);
  int32_t height = mask_blob->shape().At(2);
  CHECK_EQ(width + 2, padded_mask_blob->shape().At(1));
  CHECK_EQ(height + 2, padded_mask_blob->shape().At(0));
  CopyRegion(GetPointerPtr<T>(padded_mask_ptr, width + 2, height + 2, 1, 1), width + 2,
             GetPointerPtr<const T>(mask_ptr, width, height, 0, 0), width, width, height);
}

void ExpandRoi(BBox<uint32_t>* expanded_bbox, const Blob* rois_blob, int32_t dim0_idx,
               const Shape& mask_shape) {
  const float scale_w = (mask_shape.At(3) + 2.0) / mask_shape.At(3);
  const float scale_h = (mask_shape.At(2) + 2.0) / mask_shape.At(2);
  const auto* bbox = BBox<float>::Cast(&rois_blob->dptr<float>(dim0_idx)[1]);
  const float center_x = bbox->center_x();
  const float center_y = bbox->center_y();
  const float half_w = bbox->width() * scale_w / 2.0;
  const float half_h = bbox->height() * scale_h / 2.0;
  expanded_bbox->set_x1(center_x - half_w);
  expanded_bbox->set_y1(center_y - half_h);
  expanded_bbox->set_x2(center_x + half_w);
  expanded_bbox->set_y2(center_y + half_h);
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
void ResizeMask(cv::Mat* ret_img, Blob* padded_mask_blob, const BBox<uint32_t>* bbox) {
  cv::Mat mask_img(padded_mask_blob->shape().At(1), padded_mask_blob->shape().At(0), GetCVType<T>(),
                   padded_mask_blob->mut_dptr<T>());
  const auto& size = cv::Size(bbox->width(), bbox->height());
  cv::Mat img(size, CV_32FC1);
  cv::resize(mask_img, img, size, 0, 0, cv::INTER_LINEAR);
  *ret_img = img;
}

void BinarizeMask(std::vector<uint8_t>* binary_img, const cv::Mat& img, const float threshold) {
  binary_img->resize(img.rows * img.cols);
  CHECK(img.isContinuous());
  CHECK(img.type() == CV_32FC1);
  uint8_t* binary_img_ptr = binary_img->data();
  FOR_RANGE(int32_t, idx, 0, img.rows * img.cols) {
    binary_img_ptr[idx] = img.at<float>(idx) > threshold;
  }
}

void CopyToOutputBlob(Blob* out_blob, int32_t dim0_idx, const std::vector<uint8_t>& binary_img,
                      const BBox<uint32_t>* expanded_bbox) {
  uint8_t* im_mask_ptr = out_blob->mut_dptr<uint8_t>(dim0_idx);
  size_t width = out_blob->shape().At(2);
  size_t height = out_blob->shape().At(1);
  size_t bbox_w = expanded_bbox->width();
  size_t bbox_h = expanded_bbox->height();
  CHECK_EQ(bbox_w * bbox_h, binary_img.size());
  int32_t x0 = std::max<int32_t>(expanded_bbox->x1(), 0);
  int32_t y0 = std::max<int32_t>(expanded_bbox->y1(), 0);
  int32_t w = std::min<int32_t>(expanded_bbox->x2() + 1, width) - x0;
  int32_t h = std::min<int32_t>(expanded_bbox->y2() + 1, height) - y0;
  int32_t im_x0 = x0 - expanded_bbox->x1();
  int32_t im_y0 = y0 - expanded_bbox->y1();
  CopyRegion(GetPointerPtr<uint8_t>(im_mask_ptr, width, height, x0, y0), width,
             GetPointerPtr<const uint8_t>(binary_img.data(), bbox_w, bbox_h, im_x0, im_y0), bbox_w,
             w, h);
}

template<typename T>
void ClearBlob(Blob* blob) {
  std::memset(blob->mut_dptr<T>(), 0, blob->shape().elem_cnt() * sizeof(T));
}

}  // namespace

template<typename T>
void ImageSegmentationMaskKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const float threshold = this->op_conf().image_segmentation_mask_conf().binary_threshold();
  const Blob* mask_blob = BnInOp2Blob("masks");
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* roi_labels_blob = BnInOp2Blob("roi_labels");
  Blob* padded_mask_blob = BnInOp2Blob("padded_mask");
  Blob* out_blob = BnInOp2Blob("out");
  ClearBlob<T>(padded_mask_blob);
  ClearBlob<uint8_t>(out_blob);
  FOR_RANGE(int32_t, i, 0, mask_blob->shape().At(0)) {
    cv::Mat img;
    std::array<uint32_t, 4> expanded_bbox;
    auto* expanded_roi = BBox<uint32_t>::MutCast(&expanded_bbox[0]);

    InitPaddedMask<T>(padded_mask_blob, mask_blob, roi_labels_blob, i);
    ExpandRoi(expanded_roi, rois_blob, i, mask_blob->shape());
    ResizeMask<T>(&img, padded_mask_blob, expanded_roi);
    std::vector<uint8_t> binarized_img;
    BinarizeMask(&binarized_img, img, threshold);
    CopyToOutputBlob(out_blob, i, binarized_img, expanded_roi);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kImageSegmentationMaskConf,
                               ImageSegmentationMaskKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
