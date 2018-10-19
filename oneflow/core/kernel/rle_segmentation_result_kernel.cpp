#include <opencv2/opencv.hpp>
#include "oneflow/core/kernel/rle_segmentation_result_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/rle_util.h"

namespace oneflow {

namespace {

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

void CopyToImMask(Blob* im_mask_blob, const cv::Mat& img) { TODO(); }

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

    InitPaddedMask(padded_mask_blob, mask_blob, roi_labels_blob, i);
    ExpandRoi(expanded_roi, rois_blob, i, mask_blob->shape());
    Resize(&img, padded_mask_blob, expanded_roi);
    CopyToImMask(im_mask_blob, img);
    RleEncodeIntoOutputblob(out_blob, im_mask_blob, i);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kRleSegmentationResultConf,
                               RleSegmentationResultKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
