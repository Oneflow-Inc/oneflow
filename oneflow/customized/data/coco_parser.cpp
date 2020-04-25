#include "oneflow/customized/data/coco_parser.h"
#include "oneflow/customized/data/coco_data_reader.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

void COCOParser::Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
                       user_op::KernelComputeContext* ctx) {
  user_op::Tensor* image_tensor = ctx->Tensor4ArgNameAndIndex("image", 0);
  CHECK_NOTNULL(image_tensor);
  user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
  user_op::Tensor* bbox_tensor = ctx->Tensor4ArgNameAndIndex("gt_bbox", 0);
  user_op::Tensor* label_tensor = ctx->Tensor4ArgNameAndIndex("gt_label", 0);
  user_op::Tensor* segm_tensor = ctx->Tensor4ArgNameAndIndex("gt_segm", 0);
  user_op::Tensor* segm_offset_tensor = ctx->Tensor4ArgNameAndIndex("gt_segm_offset_mat", 0);

  MultiThreadLoop(batch_data->size(), [=](size_t i) {
    TensorBuffer* image_buffer = image_tensor->mut_dptr<TensorBuffer>() + i;
    COCOImage* image = batch_data->at(i).get();
    image_buffer->reserve(image->data.nbytes());
    std::memcpy(image_buffer->mut_data(), image->data.data(), image_buffer->nbytes());
    if (image_size_tensor) {
      auto* image_size_ptr = image_size_tensor->mut_dptr<int32_t>() + i * 2;
      image_size_ptr[0] = meta_->GetImageHeight(image->id);
      image_size_ptr[1] = meta_->GetImageWidth(image->id);
    }
    if (bbox_tensor) {
      TensorBuffer* bbox_buffer = bbox_tensor->mut_dptr<TensorBuffer>() + i;
      const auto& bbox_vec = meta_->GetBboxVec<float>(image->id);
      bbox_buffer->reserve(bbox_vec.size() * sizeof(float));
      std::copy(bbox_vec.begin(), bbox_vec.end(), bbox_buffer->mut_data<float>());
    }
    if (label_tensor) {
      TensorBuffer* label_buffer = label_tensor->mut_dptr<TensorBuffer>() + i;
      const auto& label_vec = meta_->GetLabelVec<int32_t>(image->id);
      label_buffer->reserve(label_vec.size() * sizeof(int32_t));
      std::copy(label_vec.begin(), label_vec.end(), label_buffer->mut_data<int32_t>());
    }
    if (segm_tensor && segm_offset_tensor) {
      TensorBuffer* segm_buffer = segm_tensor->mut_dptr<TensorBuffer>() + i;
      TensorBuffer* segm_offset_buffer = segm_offset_tensor->mut_dptr<TensorBuffer>() + i;
      meta_->ReadSegmentationsToTensorBuffer<float>(image->id, segm_buffer, segm_offset_buffer);
    }
  });
  // dynamic batch size
  if (batch_data->size() != image_tensor->shape().elem_cnt()) {
    CHECK_EQ(image_tensor->shape().NumAxes(), 1);
    image_tensor->mut_shape()->Set(0, batch_data->size());
  }
  if (batch_data->size() != image_size_tensor->shape().At(0)) {
    image_tensor->mut_shape()->Set(0, batch_data->size());
  }
  if (batch_data->size() != bbox_tensor->shape().elem_cnt()) {
    CHECK_EQ(bbox_tensor->shape().NumAxes(), 1);
    bbox_tensor->mut_shape()->Set(0, batch_data->size());
  }
  if (batch_data->size() != label_tensor->shape().elem_cnt()) {
    CHECK_EQ(label_tensor->shape().NumAxes(), 1);
    label_tensor->mut_shape()->Set(0, batch_data->size());
  }
  if (batch_data->size() != segm_tensor->shape().elem_cnt()) {
    CHECK_EQ(segm_tensor->shape().NumAxes(), 1);
    CHECK_EQ(segm_offset_tensor->shape().NumAxes(), 1);
    CHECK_EQ(segm_tensor->shape().elem_cnt(), segm_offset_tensor->shape().elem_cnt());
    segm_tensor->mut_shape()->Set(0, batch_data->size());
    segm_offset_tensor->mut_shape()->Set(0, batch_data->size());
  }
}

}  // namespace oneflow
