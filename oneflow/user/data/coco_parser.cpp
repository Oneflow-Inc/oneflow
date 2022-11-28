/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/data/coco_parser.h"
#include "oneflow/user/data/coco_data_reader.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {
namespace data {

void COCOParser::Parse(BatchType& batch_data, user_op::KernelComputeContext* ctx) {
  user_op::Tensor* image_tensor = ctx->Tensor4ArgNameAndIndex("image", 0);
  CHECK_NOTNULL(image_tensor);
  user_op::Tensor* image_id_tensor = ctx->Tensor4ArgNameAndIndex("image_id", 0);
  user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
  user_op::Tensor* bbox_tensor = ctx->Tensor4ArgNameAndIndex("gt_bbox", 0);
  user_op::Tensor* label_tensor = ctx->Tensor4ArgNameAndIndex("gt_label", 0);
  user_op::Tensor* segm_tensor = ctx->Tensor4ArgNameAndIndex("gt_segm", 0);
  user_op::Tensor* segm_index_tensor = ctx->Tensor4ArgNameAndIndex("gt_segm_index", 0);

  MultiThreadLoop(batch_data.size(), [&](size_t i) {
    TensorBuffer* image_buffer = image_tensor->mut_dptr<TensorBuffer>() + i;
    COCOImage& image = batch_data[i];
    image_buffer->Swap(image.data);
    if (image_size_tensor) {
      auto* image_size_ptr = image_size_tensor->mut_dptr<int32_t>() + i * 2;
      image_size_ptr[0] = meta_->GetImageHeight(image.index);
      image_size_ptr[1] = meta_->GetImageWidth(image.index);
    }
    if (image_id_tensor) {
      auto* image_id_ptr = image_id_tensor->mut_dptr<int64_t>();
      image_id_ptr[i] = image.id;
    }
    if (bbox_tensor) {
      TensorBuffer* bbox_buffer = bbox_tensor->mut_dptr<TensorBuffer>() + i;
      const auto& bbox_vec = meta_->GetBboxVec<float>(image.index);
      CHECK_EQ(bbox_vec.size() % 4, 0);
      int64_t num_bboxes = bbox_vec.size() / 4;
      bbox_buffer->Resize(Shape({num_bboxes, 4}), DataType::kFloat);
      std::copy(bbox_vec.begin(), bbox_vec.end(), bbox_buffer->mut_data<float>());
    }
    if (label_tensor) {
      TensorBuffer* label_buffer = label_tensor->mut_dptr<TensorBuffer>() + i;
      const auto& label_vec = meta_->GetLabelVec<int32_t>(image.index);
      label_buffer->Resize(Shape({static_cast<int64_t>(label_vec.size())}), DataType::kInt32);
      std::copy(label_vec.begin(), label_vec.end(), label_buffer->mut_data<int32_t>());
    }
    if (segm_tensor && segm_index_tensor) {
      TensorBuffer* segm_buffer = segm_tensor->mut_dptr<TensorBuffer>() + i;
      TensorBuffer* segm_index_buffer = segm_index_tensor->mut_dptr<TensorBuffer>() + i;
      meta_->ReadSegmentationsToTensorBuffer<float>(image.index, segm_buffer, segm_index_buffer);
    }
  });
  // dynamic batch size
  if (image_tensor->shape_view().elem_cnt() != batch_data.size()) {
    CHECK_EQ(image_tensor->shape_view().NumAxes(), 1);
    image_tensor->mut_shape_view().Set(0, batch_data.size());
  }
  if (image_id_tensor && image_id_tensor->shape_view().At(0) != batch_data.size()) {
    image_id_tensor->mut_shape_view().Set(0, batch_data.size());
  }
  if (image_size_tensor && image_size_tensor->shape_view().At(0) != batch_data.size()) {
    image_size_tensor->mut_shape_view().Set(0, batch_data.size());
  }
  if (bbox_tensor && bbox_tensor->shape_view().elem_cnt() != batch_data.size()) {
    CHECK_EQ(bbox_tensor->shape_view().NumAxes(), 1);
    bbox_tensor->mut_shape_view().Set(0, batch_data.size());
  }
  if (label_tensor && label_tensor->shape_view().elem_cnt() != batch_data.size()) {
    CHECK_EQ(label_tensor->shape_view().NumAxes(), 1);
    label_tensor->mut_shape_view().Set(0, batch_data.size());
  }
  if (segm_tensor && segm_index_tensor
      && segm_tensor->shape_view().elem_cnt() != batch_data.size()) {
    CHECK_EQ(segm_tensor->shape_view().NumAxes(), 1);
    CHECK_EQ(segm_index_tensor->shape_view().NumAxes(), 1);
    CHECK_EQ(segm_tensor->shape_view().elem_cnt(), segm_index_tensor->shape_view().elem_cnt());
    segm_tensor->mut_shape_view().Set(0, batch_data.size());
    segm_index_tensor->mut_shape_view().Set(0, batch_data.size());
  }
}

}  // namespace data
}  // namespace oneflow
