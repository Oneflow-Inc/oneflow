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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

struct CropWindow {
  float x;
  float y;
  float w;
  float h;

  CropWindow() : x(0.0f), y(0.0f), w(0.0f), h(0.0f) {}
  CropWindow(float _x, float _y, float _w, float _h) : x(_x), y(_y), w(_w), h(_h) {}

  template<typename T = float>
  T left() const {
    return static_cast<T>(x);
  }
  template<typename T = float>
  T right() const {
    return static_cast<T>(x + w);
  }
  template<typename T = float>
  T top() const {
    return static_cast<T>(y);
  }
  template<typename T = float>
  T bottom() const {
    return static_cast<T>(y + h);
  }
  template<typename T = float>
  T width() const {
    return static_cast<T>(w);
  }
  template<typename T = float>
  T height() const {
    return static_cast<T>(h);
  }
  template<typename T = float>
  T area() const {
    return width<T>() * height<T>();
  }
  template<typename T = float>
  bool cover(T x, T y) const {
    return x >= left<T>() && x <= right<T>() && y >= top<T>() && y <= bottom<T>();
  }
};

using IouOverlapRange = std::pair<float, float>;
using ShrinkRateRange = std::pair<float, float>;
using SizeShrinkRateRange = std::pair<ShrinkRateRange, ShrinkRateRange>;
using AspectRatioRange = std::pair<float, float>;

bool CalcIOUAndCheckInRange(const CropWindow& crop, const TensorBuffer& bbox,
                            const IouOverlapRange& iou_range) {
  CHECK_EQ(bbox.shape().NumAxes(), 2);
  CHECK_EQ(bbox.shape().At(1), 4);
  CHECK_EQ(bbox.data_type(), DataType::kFloat);
  int num_boxes = bbox.shape().At(0);
  FOR_RANGE(int, i, 0, num_boxes) {
    const float* bbox_ptr = bbox.data<float>() + i * 4;
    float inter_w = std::min(bbox_ptr[2], crop.right()) - std::max(bbox_ptr[0], crop.left());
    float inter_h = std::min(bbox_ptr[3], crop.bottom()) - std::max(bbox_ptr[1], crop.top());
    float inter_area = 0.0f;
    if (inter_w > 0.0f && inter_h > 0.0f) { inter_area = inter_w * inter_h; }
    float box_area = (bbox_ptr[2] - bbox_ptr[0]) * (bbox_ptr[3] - bbox_ptr[1]);
    float crop_area = crop.area();
    CHECK_GT(box_area, 0.0f);
    CHECK_GT(crop_area, 0.0f);
    float iou = inter_area / (box_area + crop_area - inter_area);
    if (iou >= iou_range.first && iou <= iou_range.second) { return true; }
  }
  return false;
}

bool CheckBBoxCenterLocatedInCropWindow(const CropWindow& crop, const TensorBuffer& bbox,
                                        TensorBuffer* keep_mask) {
  CHECK_EQ(bbox.shape().NumAxes(), 2);
  CHECK_EQ(bbox.shape().At(1), 4);
  CHECK_EQ(bbox.data_type(), DataType::kFloat);
  int num_boxes = bbox.shape().At(0);
  bool has_box_located_in_crop = false;
  keep_mask->Resize(Shape({num_boxes}), DataType::kInt8);
  FOR_RANGE(int, i, 0, num_boxes) {
    const float* bbox_ptr = bbox.data<float>() + i * 4;
    float center_x = (bbox_ptr[0] + bbox_ptr[2]) / 2.0f;
    float center_y = (bbox_ptr[1] + bbox_ptr[3]) / 2.0f;
    if (crop.cover(center_x, center_y)) {
      has_box_located_in_crop = true;
      keep_mask->mut_data<int8_t>()[i] = 1;
    }
  }
  return has_box_located_in_crop;
}

class SSDRandomCropGenerator final : public user_op::OpKernelState {
 public:
  SSDRandomCropGenerator(const std::vector<float>& min_iou_vec,
                         const std::vector<float>& max_iou_vec,
                         const SizeShrinkRateRange& shrink_rate_range,
                         const AspectRatioRange& aspect_ratio_range, int64_t seed)
      : seed_(seed),
        gen_(seed),
        size_shrink_rates_(shrink_rate_range),
        aspect_ratios_(aspect_ratio_range) {
    CHECK_EQ(min_iou_vec.size(), max_iou_vec.size());
    FOR_RANGE(size_t, i, 0, min_iou_vec.size()) {
      iou_overlaps_.emplace_back(std::make_pair(min_iou_vec.at(i), max_iou_vec.at(i)));
    }
  }
  ~SSDRandomCropGenerator() = default;

  IouOverlapRange RandomChoiceIOURange() {
    auto iou_it = RandomChoice(iou_overlaps_.begin(), iou_overlaps_.end());
    return *iou_it;
  }

  bool RandomCropWindow(int64_t width, int64_t height, CropWindow* crop) {
    float new_width = RandomUniform(std::make_pair(width * size_shrink_rates_.first.first,
                                                   width * size_shrink_rates_.first.second));
    float new_height = RandomUniform(std::make_pair(height * size_shrink_rates_.second.first,
                                                    height * size_shrink_rates_.second.second));
    float new_aspect_ratio = new_height / new_width;
    if (new_aspect_ratio >= aspect_ratios_.first && new_aspect_ratio <= aspect_ratios_.second) {
      float new_left = RandomUniform(std::make_pair(0.0f, width - new_width));
      float new_top = RandomUniform(std::make_pair(0.0f, height - new_height));
      *crop = {new_left, new_top, new_width, new_height};
      return true;
    }
    return false;
  }

 protected:
  template<typename Iter>
  Iter RandomChoice(Iter start, Iter end) {
    CHECK(start != end);
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(gen_));
    return start;
  }

  template<typename T>
  T RandomUniform(const std::pair<T, T>& range) {
    CHECK_LE(range.first, range.second);
    if (range.first == range.second) { return range.first; }
    std::uniform_real_distribution<T> dis(range.first, range.second);
    return dis(gen_);
  }

 private:
  int64_t seed_;
  std::mt19937 gen_;
  std::vector<IouOverlapRange> iou_overlaps_;
  SizeShrinkRateRange size_shrink_rates_;
  AspectRatioRange aspect_ratios_;
};

class SSDRandomCropKernelState final : public user_op::OpKernelState {
 public:
  explicit SSDRandomCropKernelState(size_t size) : generators_(size) {}
  ~SSDRandomCropKernelState() = default;

  SSDRandomCropGenerator* Get(int idx) { return generators_.at(idx).get(); }

  void New(int idx, const std::vector<float>& min_iou_vec, const std::vector<float>& max_iou_vec,
           const SizeShrinkRateRange& shrink_rate_range, const AspectRatioRange& aspect_ratio_range,
           int64_t seed) {
    CHECK_LT(idx, generators_.size());
    generators_.at(idx).reset(new SSDRandomCropGenerator(
        min_iou_vec, max_iou_vec, shrink_rate_range, aspect_ratio_range, seed));
  }

 private:
  std::vector<std::unique_ptr<SSDRandomCropGenerator>> generators_;
};

void CropImage(const TensorBuffer& origin_image, const CropWindow& crop, TensorBuffer* image) {
  CHECK_EQ(origin_image.shape().NumAxes(), 3);
  int64_t channels = origin_image.shape().At(2);
  image->Resize(Shape({crop.height<int>(), crop.width<int>(), channels}), origin_image.data_type());
  cv::Mat origin_image_mat = GenCvMat4ImageBuffer(origin_image);
  cv::Mat crop_image_mat = GenCvMat4ImageBuffer(*image);
  cv::Rect crop_roi = {crop.left<int>(), crop.top<int>(), crop.width<int>(), crop.height<int>()};
  origin_image_mat(crop_roi).copyTo(crop_image_mat);
  CHECK(crop_image_mat.isContinuous());
  CHECK_EQ(crop_image_mat.ptr(), image->data());
  CHECK_EQ(crop_image_mat.rows, image->shape().At(0));
  CHECK_EQ(crop_image_mat.cols, image->shape().At(1));
  CHECK_EQ(crop_image_mat.channels(), channels);
}

void CropBBox(const TensorBuffer& origin_bbox, const CropWindow& crop,
              const TensorBuffer& keep_mask, TensorBuffer* bbox) {
  CHECK_EQ(origin_bbox.shape().NumAxes(), 2);
  CHECK_EQ(origin_bbox.shape().At(1), 4);
  CHECK_EQ(origin_bbox.data_type(), DataType::kFloat);
  int64_t num_boxes = origin_bbox.shape().At(0);
  CHECK_EQ(keep_mask.shape().NumAxes(), 1);
  CHECK_EQ(keep_mask.shape().At(0), num_boxes);
  CHECK_EQ(keep_mask.data_type(), DataType::kInt8);
  int64_t num_keeps = 0;
  FOR_RANGE(int, i, 0, num_boxes) {
    if (static_cast<bool>(keep_mask.data<int8_t>()[i])) { num_keeps += 1; }
  }
  bbox->Resize(Shape({num_keeps, 4}), origin_bbox.data_type());
  num_keeps = 0;
  FOR_RANGE(int, i, 0, num_boxes) {
    if (static_cast<bool>(keep_mask.data<int8_t>()[i])) {
      const float* origin_bbox_ptr = origin_bbox.data<float>() + i * 4;
      float* bbox_ptr = bbox->mut_data<float>() + num_keeps * 4;
      bbox_ptr[0] = std::max(origin_bbox_ptr[0], crop.left()) - crop.left();
      bbox_ptr[1] = std::max(origin_bbox_ptr[1], crop.top()) - crop.top();
      float bbox_right = std::min(origin_bbox_ptr[2], crop.right()) - crop.left();
      bbox_ptr[2] = bbox_right <= crop.width<int>() ? bbox_right : static_cast<int>(bbox_right);
      float bbox_bottom = std::min(origin_bbox_ptr[3], crop.bottom()) - crop.top();
      bbox_ptr[3] = bbox_bottom <= crop.height<int>() ? bbox_bottom : static_cast<int>(bbox_bottom);
      num_keeps += 1;
    }
  }
  CHECK_EQ(num_keeps, bbox->shape().At(0));
}

void FilterLabels(const TensorBuffer& origin_label, const TensorBuffer& keep_mask,
                  TensorBuffer* label) {
  CHECK_EQ(origin_label.shape().NumAxes(), 1);
  CHECK_EQ(origin_label.data_type(), DataType::kInt32);
  int64_t num_labels = origin_label.shape().At(0);
  CHECK_EQ(keep_mask.shape().NumAxes(), 1);
  CHECK_EQ(keep_mask.shape().At(0), num_labels);
  CHECK_EQ(keep_mask.data_type(), DataType::kInt8);
  int64_t num_keeps = 0;
  FOR_RANGE(int, i, 0, num_labels) {
    if (static_cast<bool>(keep_mask.data<int8_t>()[i])) { num_keeps += 1; }
  }
  label->Resize(Shape({num_keeps}), origin_label.data_type());
  num_keeps = 0;
  FOR_RANGE(int, i, 0, num_labels) {
    if (static_cast<bool>(keep_mask.data<int8_t>()[i])) {
      label->mut_data<int32_t>()[num_keeps] = origin_label.data<int32_t>()[i];
      num_keeps += 1;
    }
  }
  CHECK_EQ(num_keeps, label->shape().At(0));
}

class SSDRandomCropKernel final : public user_op::OpKernel {
 public:
  SSDRandomCropKernel() = default;
  ~SSDRandomCropKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& min_iou_vec = ctx->Attr<std::vector<float>>("min_iou_overlaps");
    const auto& max_iou_vec = ctx->Attr<std::vector<float>>("max_iou_overlaps");
    float min_width_shrink_rate = ctx->Attr<float>("min_width_shrink_rate");
    float max_width_shrink_rate = ctx->Attr<float>("max_width_shrink_rate");
    float min_height_shrink_rate = ctx->Attr<float>("min_height_shrink_rate");
    float max_height_shrink_rate = ctx->Attr<float>("max_height_shrink_rate");
    float min_crop_aspect_ratio = ctx->Attr<float>("min_crop_aspect_ratio");
    float max_crop_aspect_ratio = ctx->Attr<float>("max_crop_aspect_ratio");

    const user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out_image", 0);
    CHECK_EQ(out_desc->shape().NumAxes(), 1);
    int64_t batch_size = out_desc->shape().At(0);
    CHECK_GT(batch_size, 0);
    int64_t seed = GetOpKernelRandomSeed(ctx);
    std::seed_seq seq{seed};
    std::vector<int> seeds(batch_size);
    seq.generate(seeds.begin(), seeds.end());

    auto* kernel_state = new SSDRandomCropKernelState(batch_size);
    FOR_RANGE(int, i, 0, batch_size) {
      kernel_state->New(i, min_iou_vec, max_iou_vec,
                        {{min_width_shrink_rate, max_width_shrink_rate},
                         {min_height_shrink_rate, max_height_shrink_rate}},
                        {min_crop_aspect_ratio, max_crop_aspect_ratio}, seeds.at(i));
    }
    return std::shared_ptr<SSDRandomCropKernelState>(kernel_state);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* image_tensor = ctx->Tensor4ArgNameAndIndex("image", 0);
    const user_op::Tensor* bbox_tensor = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* label_tensor = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* out_image_tensor = ctx->Tensor4ArgNameAndIndex("out_image", 0);
    user_op::Tensor* out_bbox_tensor = ctx->Tensor4ArgNameAndIndex("out_bbox", 0);
    user_op::Tensor* out_label_tensor = ctx->Tensor4ArgNameAndIndex("out_label", 0);
    CHECK_EQ(image_tensor->shape().elem_cnt(), out_image_tensor->shape().elem_cnt());
    int64_t batch_size = image_tensor->shape().elem_cnt();
    if (bbox_tensor) {
      CHECK_NOTNULL(out_bbox_tensor);
      CHECK_EQ(bbox_tensor->shape().elem_cnt(), out_bbox_tensor->shape().elem_cnt());
      CHECK_EQ(bbox_tensor->shape().elem_cnt(), batch_size);
    }
    if (label_tensor) {
      CHECK_NOTNULL(out_label_tensor);
      CHECK_EQ(label_tensor->shape().elem_cnt(), out_label_tensor->shape().elem_cnt());
      CHECK_EQ(label_tensor->shape().elem_cnt(), batch_size);
    }

    auto IdenticalOutput = [&](int i) {
      out_image_tensor->mut_dptr<TensorBuffer>()[i].CopyFrom(image_tensor->dptr<TensorBuffer>()[i]);
      if (bbox_tensor) {
        out_bbox_tensor->mut_dptr<TensorBuffer>()[i].CopyFrom(bbox_tensor->dptr<TensorBuffer>()[i]);
      }
      if (label_tensor) {
        out_label_tensor->mut_dptr<TensorBuffer>()[i].CopyFrom(
            label_tensor->dptr<TensorBuffer>()[i]);
      }
    };
    auto* kernel_state = dynamic_cast<SSDRandomCropKernelState*>(state);

    MultiThreadLoop(batch_size, [&](size_t i) {
      const TensorBuffer& image_buffer = image_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(image_buffer.shape().NumAxes(), 3);
      const int64_t image_height = image_buffer.shape().At(0);
      const int64_t image_width = image_buffer.shape().At(1);

      SSDRandomCropGenerator* generator = kernel_state->Get(i);
      std::pair<float, float> iou_range = generator->RandomChoiceIOURange();
      if (iou_range.first < 0.0f || iou_range.second < 0.0f) {
        IdenticalOutput(i);
      } else {
        CropWindow crop;
        TensorBuffer keep_mask;
        const TensorBuffer* bbox_buffer_ptr =
            bbox_tensor ? bbox_tensor->dptr<TensorBuffer>() : nullptr;
        int32_t attempt_count = ctx->Attr<int32_t>("max_num_attempts");
        do {
          if (!generator->RandomCropWindow(image_width, image_height, &crop)) { continue; }
          if (bbox_buffer_ptr) {
            if (!CalcIOUAndCheckInRange(crop, bbox_buffer_ptr[i], iou_range)) { continue; }
            if (!CheckBBoxCenterLocatedInCropWindow(crop, bbox_buffer_ptr[i], &keep_mask)) {
              continue;
            }
          }
          break;
        } while (--attempt_count > 0);

        if (attempt_count == 0) {
          IdenticalOutput(i);
        } else {
          CropImage(image_buffer, crop, out_image_tensor->mut_dptr<TensorBuffer>() + i);
          if (bbox_tensor) {
            CropBBox(bbox_buffer_ptr[i], crop, keep_mask,
                     out_bbox_tensor->mut_dptr<TensorBuffer>() + i);
          }
          if (label_tensor) {
            FilterLabels(label_tensor->dptr<TensorBuffer>()[i], keep_mask,
                         out_label_tensor->mut_dptr<TensorBuffer>() + i);
          }
        }
      }
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

Maybe<void> ProposeInplace(const user_op::InferContext& ctx,
                           user_op::AddInplaceArgPair AddInplaceArgPairFn) {
  OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out_image", 0, "image", 0, true));
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_KERNEL("ssd_random_crop")
    .SetCreateFn<SSDRandomCropKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     & (user_op::HobDataType("image", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out_image", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(ProposeInplace);
;
}  // namespace oneflow
