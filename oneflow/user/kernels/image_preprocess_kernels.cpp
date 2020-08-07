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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/image/image_util.h"
#include "oneflow/user/kernels/random_crop_kernel_state.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

int GetOpencvInterp(const std::string& interp_type) {
  if (interp_type == "Linear") {
    return cv::INTER_LINEAR;
  } else if (interp_type == "NN") {
    return cv::INTER_NEAREST;
  } else if (interp_type == "Cubic") {
    return cv::INTER_CUBIC;
  } else {
    UNIMPLEMENTED();
    return -1;
  }
}

}  // namespace

class ResizeToStaticShapeKernel final : public user_op::OpKernel {
 public:
  ResizeToStaticShapeKernel() = default;
  ~ResizeToStaticShapeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = in_blob->shape().At(0);
    CHECK_GT(record_num, 0);

    const TensorBuffer* buffers = in_blob->dptr<TensorBuffer>();
    uint8_t* out_dptr = out_blob->mut_dptr<uint8_t>();
    const ShapeView& out_shape = out_blob->shape();
    CHECK(out_shape.NumAxes() == 4);  // {N, H, W, C}
    int rsz_h = out_shape.At(1);
    int rsz_w = out_shape.At(2);
    int C = out_shape.At(3);
    CHECK(C == 3 || C == 1);
    int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
    const std::string& interp_type = ctx->Attr<std::string>("interp_type");
    int64_t one_sample_elem_cnt = rsz_h * rsz_w * C;
    int opencv_inter_type = GetOpencvInterp(interp_type);

    MultiThreadLoop(record_num, [&](size_t i) {
      const TensorBuffer* buffer = buffers + i;
      uint8_t* dptr = out_dptr + one_sample_elem_cnt * i;
      const Shape& in_shape = buffer->shape();
      CHECK(in_shape.NumAxes() == 3);  // {H, W, C}
      int H = in_shape.At(0);
      int W = in_shape.At(1);
      CHECK_EQ(C, in_shape.At(2));
      const cv::Mat image = CreateMatWithPtr(H, W, channel_flag, buffer->data<uint8_t>());
      cv::Mat rsz_image = CreateMatWithPtr(rsz_h, rsz_w, channel_flag, dptr);
      cv::resize(image, rsz_image, cv::Size(rsz_w, rsz_h), 0, 0, opencv_inter_type);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("image_resize")
    .SetCreateFn<ResizeToStaticShapeKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kUInt8));

class ResizeShorterToTensorBufferKernel final : public user_op::OpKernel {
 public:
  ResizeShorterToTensorBufferKernel() = default;
  ~ResizeShorterToTensorBufferKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = in_blob->shape().At(0);
    CHECK_GT(record_num, 0);

    const TensorBuffer* in_buffers = in_blob->dptr<TensorBuffer>();
    TensorBuffer* out_buffers = out_blob->mut_dptr<TensorBuffer>();
    int64_t resize_shorter = ctx->Attr<int64_t>("resize_shorter");

    MultiThreadLoop(record_num, [&](size_t i) {
      const TensorBuffer* in_buffer = in_buffers + i;
      TensorBuffer* out_buffer = out_buffers + i;
      const Shape& in_shape = in_buffer->shape();
      CHECK_EQ(in_shape.NumAxes(), 3);  // {H, W, C}
      int64_t H = in_shape.At(0);
      int64_t W = in_shape.At(1);
      int64_t C = in_shape.At(2);
      CHECK(C == 3 || C == 1);
      int64_t rsz_h = resize_shorter;
      int64_t rsz_w = resize_shorter;
      if (H < W) {
        rsz_w = resize_shorter * (static_cast<float>(W) / static_cast<float>(H));
      } else {
        rsz_h = resize_shorter * (static_cast<float>(H) / static_cast<float>(W));
      }
      Shape out_shape({rsz_h, rsz_w, C});
      out_buffer->Resize(out_shape, DataType::kUInt8);
      int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
      const std::string& interp_type = ctx->Attr<std::string>("interp_type");
      int opencv_inter_type = GetOpencvInterp(interp_type);

      const cv::Mat image = CreateMatWithPtr(H, W, channel_flag, in_buffer->data<uint8_t>());
      cv::Mat rsz_image = CreateMatWithPtr(rsz_h, rsz_w, channel_flag, out_buffer->data<uint8_t>());
      cv::resize(image, rsz_image, cv::Size(rsz_w, rsz_h), 0, 0, opencv_inter_type);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("image_resize")
    .SetCreateFn<ResizeShorterToTensorBufferKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

namespace {

enum TensorLayout {
  kNCHW = 0,
  kNHWC = 1,
};

template<TensorLayout layout>
inline int64_t GetOffset(int64_t h, int64_t w, int64_t c, int64_t H, int64_t W, int64_t C);

template<>
inline int64_t GetOffset<TensorLayout::kNCHW>(int64_t h, int64_t w, int64_t c, int64_t H, int64_t W,
                                              int64_t C) {
  return c * H * W + h * W + w;  // C, H, W
}

template<>
inline int64_t GetOffset<TensorLayout::kNHWC>(int64_t h, int64_t w, int64_t c, int64_t H, int64_t W,
                                              int64_t C) {
  return h * W * C + w * C + c;  // H, W, C
}

template<bool mirror>
inline int64_t GetInputW(int64_t out_w, int64_t out_W, int64_t in_W, float crop_pos_x);

template<>
inline int64_t GetInputW<true>(int64_t out_w, int64_t out_W, int64_t in_W, float crop_pos_x) {
  return (in_W - out_W) * crop_pos_x + (out_W - 1 - out_w);
}

template<>
inline int64_t GetInputW<false>(int64_t out_w, int64_t out_W, int64_t in_W, float crop_pos_x) {
  return (in_W - out_W) * crop_pos_x + out_w;
}

template<TensorLayout output_layout, bool mirror>
void CMN1Sample(int64_t C, int64_t in_H, int64_t in_W, int64_t out_H, int64_t out_W,
                float crop_pos_y, float crop_pos_x, const uint8_t* in_dptr, float* out_dptr,
                const std::vector<float>& mean_vec, const std::vector<float>& inv_std_vec) {
  CHECK_LE(out_H, in_H);
  CHECK_LE(out_W, in_W);
  for (int64_t c = 0; c < C; ++c) {
    float mean = mean_vec.at(c);
    float inv_std = inv_std_vec.at(c);
    for (int64_t out_h = 0; out_h < out_H; ++out_h) {
      int64_t in_h = (in_H - out_H) * crop_pos_y + out_h;
      for (int64_t out_w = 0; out_w < out_W; ++out_w) {
        int64_t in_w = GetInputW<mirror>(out_w, out_W, in_W, crop_pos_x);
        int64_t in_offset = GetOffset<TensorLayout::kNHWC>(in_h, in_w, c, in_H, in_W, C);
        int64_t out_offset = GetOffset<output_layout>(out_h, out_w, c, out_H, out_W, C);
        out_dptr[out_offset] = (static_cast<float>(in_dptr[in_offset]) - mean) * inv_std;
      }
    }
  }
}

std::vector<int8_t> GetMirrorVec(user_op::KernelComputeContext* ctx) {
  std::vector<int8_t> mirror;
  user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
  user_op::Tensor* mirror_blob = ctx->Tensor4ArgNameAndIndex("mirror", 0);
  int64_t record_num = in_blob->shape().At(0);
  if (mirror_blob) {
    CHECK_EQ(record_num, mirror_blob->shape().elem_cnt());
    mirror.insert(mirror.end(), mirror_blob->dptr<int8_t>(),
                  mirror_blob->dptr<int8_t>() + record_num);
  } else {
    mirror.resize(record_num, 0);
  }
  return mirror;
}

class CMNAttr final : public user_op::OpKernelState {
 public:
  CMNAttr(user_op::KernelInitContext* ctx) {
    mean_vec_ = ctx->Attr<std::vector<float>>("mean");
    const std::vector<float>& std_vec = ctx->Attr<std::vector<float>>("std");
    const std::string& color_space = ctx->Attr<std::string>("color_space");
    int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
    CHECK(mean_vec_.size() == 1 || mean_vec_.size() == C);
    CHECK(std_vec.size() == 1 || std_vec.size() == C);
    for (float elem : std_vec) { inv_std_vec_.push_back(1.0f / elem); }
    if (mean_vec_.size() == 1) { mean_vec_.resize(C, mean_vec_.at(0)); }
    if (inv_std_vec_.size() == 1) { inv_std_vec_.resize(C, inv_std_vec_.at(0)); }
  }
  ~CMNAttr() = default;

  const std::vector<float>& mean_vec() const { return mean_vec_; }
  const std::vector<float>& inv_std_vec() const { return inv_std_vec_; }

 private:
  std::vector<float> mean_vec_;
  std::vector<float> inv_std_vec_;
};

}  // namespace

class CropMirrorNormalizeFromStaticShapeToFloatKernel final : public user_op::OpKernel {
 public:
  CropMirrorNormalizeFromStaticShapeToFloatKernel() = default;
  ~CropMirrorNormalizeFromStaticShapeToFloatKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<CMNAttr>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* cmn_attr = dynamic_cast<CMNAttr*>(state);
    const std::vector<float>& mean_vec = cmn_attr->mean_vec();
    const std::vector<float>& inv_std_vec = cmn_attr->inv_std_vec();
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    std::vector<int8_t> mirror = GetMirrorVec(ctx);
    int64_t record_num = in_blob->shape().At(0);
    const std::string& color_space = ctx->Attr<std::string>("color_space");
    int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
    float crop_pos_y = ctx->Attr<float>("crop_pos_y");
    float crop_pos_x = ctx->Attr<float>("crop_pos_x");
    const std::string& output_layout = ctx->Attr<std::string>("output_layout");
    float* out_dptr = out_blob->mut_dptr<float>();

    const uint8_t* in_dptr = in_blob->dptr<uint8_t>();
    const ShapeView& in_shape = in_blob->shape();
    int64_t N = in_shape.At(0);
    int64_t in_H = in_shape.At(1);
    int64_t in_W = in_shape.At(2);
    CHECK_EQ(C, in_shape.At(3));
    int64_t in_image_elem_cnt = in_H * in_W * C;
    const ShapeView& out_shape = out_blob->shape();
    CHECK_EQ(out_shape.NumAxes(), 4);
    CHECK_EQ(out_shape.At(0), N);
    if (output_layout == "NCHW") {
      CHECK_EQ(out_shape.At(1), C);
      int64_t out_H = out_shape.At(2);
      int64_t out_W = out_shape.At(3);
      int64_t out_image_elem_cnt = C * out_H * out_W;
      MultiThreadLoop(record_num, [&](size_t i) {
        if (mirror.at(i)) {
          CMN1Sample<TensorLayout::kNCHW, true>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_dptr + in_image_elem_cnt * i,
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        } else {
          CMN1Sample<TensorLayout::kNCHW, false>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_dptr + in_image_elem_cnt * i,
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        }
      });
    } else if (output_layout == "NHWC") {
      CHECK_EQ(out_shape.At(3), C);
      int64_t out_H = out_shape.At(1);
      int64_t out_W = out_shape.At(2);
      int64_t out_image_elem_cnt = C * out_H * out_W;
      MultiThreadLoop(record_num, [&](size_t i) {
        if (mirror.at(i)) {
          CMN1Sample<TensorLayout::kNHWC, true>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_dptr + in_image_elem_cnt * i,
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        } else {
          CMN1Sample<TensorLayout::kNHWC, false>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_dptr + in_image_elem_cnt * i,
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        }
      });
    } else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("crop_mirror_normalize_from_uint8")
    .SetCreateFn<CropMirrorNormalizeFromStaticShapeToFloatKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kUInt8)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

class CropMirrorNormalizeFromTensorBufferToFloatKernel final : public user_op::OpKernel {
 public:
  CropMirrorNormalizeFromTensorBufferToFloatKernel() = default;
  ~CropMirrorNormalizeFromTensorBufferToFloatKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<CMNAttr>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* cmn_attr = dynamic_cast<CMNAttr*>(state);
    const std::vector<float>& mean_vec = cmn_attr->mean_vec();
    const std::vector<float>& inv_std_vec = cmn_attr->inv_std_vec();
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    std::vector<int8_t> mirror = GetMirrorVec(ctx);
    int64_t record_num = in_blob->shape().At(0);
    const std::string& color_space = ctx->Attr<std::string>("color_space");
    int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
    float crop_pos_y = ctx->Attr<float>("crop_pos_y");
    float crop_pos_x = ctx->Attr<float>("crop_pos_x");
    const std::string& output_layout = ctx->Attr<std::string>("output_layout");
    float* out_dptr = out_blob->mut_dptr<float>();

    const TensorBuffer* in_buffers = in_blob->dptr<TensorBuffer>();
    const ShapeView& in_shape = in_blob->shape();
    int64_t N = in_shape.At(0);
    CHECK_EQ(in_shape.NumAxes(), 1);
    const ShapeView& out_shape = out_blob->shape();
    CHECK_EQ(out_shape.NumAxes(), 4);
    CHECK_EQ(out_shape.At(0), N);
    if (output_layout == "NCHW") {
      CHECK_EQ(out_shape.At(1), C);
      int64_t out_H = out_shape.At(2);
      int64_t out_W = out_shape.At(3);
      int64_t out_image_elem_cnt = C * out_H * out_W;
      MultiThreadLoop(record_num, [&](size_t i) {
        const TensorBuffer* in_buffer = in_buffers + i;
        const Shape& in_shape = in_buffer->shape();
        CHECK_EQ(in_shape.NumAxes(), 3);  // H, W, C
        int64_t in_H = in_shape.At(0);
        int64_t in_W = in_shape.At(1);
        CHECK_EQ(C, in_shape.At(2));
        if (mirror.at(i)) {
          CMN1Sample<TensorLayout::kNCHW, true>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_buffer->data<uint8_t>(),
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        } else {
          CMN1Sample<TensorLayout::kNCHW, false>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_buffer->data<uint8_t>(),
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        }
      });
    } else if (output_layout == "NHWC") {
      CHECK_EQ(out_shape.At(3), C);
      int64_t out_H = out_shape.At(1);
      int64_t out_W = out_shape.At(2);
      int64_t out_image_elem_cnt = C * out_H * out_W;
      MultiThreadLoop(record_num, [&](size_t i) {
        const TensorBuffer* in_buffer = in_buffers + i;
        const Shape& in_shape = in_buffer->shape();
        CHECK_EQ(in_shape.NumAxes(), 3);  // H, W, C
        int64_t in_H = in_shape.At(0);
        int64_t in_W = in_shape.At(1);
        CHECK_EQ(C, in_shape.At(2));
        if (mirror.at(i)) {
          CMN1Sample<TensorLayout::kNHWC, true>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_buffer->data<uint8_t>(),
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        } else {
          CMN1Sample<TensorLayout::kNHWC, false>(
              C, in_H, in_W, out_H, out_W, crop_pos_y, crop_pos_x, in_buffer->data<uint8_t>(),
              out_dptr + out_image_elem_cnt * i, mean_vec, inv_std_vec);
        }
      });
    } else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("crop_mirror_normalize_from_tensorbuffer")
    .SetCreateFn<CropMirrorNormalizeFromTensorBufferToFloatKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

namespace {

class RandBoolGen final : public user_op::OpKernelState {
 public:
  explicit RandBoolGen(float prob, int64_t seed) : dis_(prob), rng_(seed) {}
  ~RandBoolGen() = default;

  bool GetNextBool() { return dis_(rng_); }

 private:
  std::bernoulli_distribution dis_;
  std::mt19937 rng_;
};

}  // namespace

class CoinFlipKernel final : public user_op::OpKernel {
 public:
  CoinFlipKernel() = default;
  ~CoinFlipKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    float prob = ctx->Attr<float>("probability");
    int64_t seed = GetOpKernelRandomSeed(ctx);
    std::shared_ptr<RandBoolGen> rand_bool_gen(new RandBoolGen(prob, seed));
    return rand_bool_gen;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* rand_bool_gen = dynamic_cast<RandBoolGen*>(state);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int8_t* dptr = out_blob->mut_dptr<int8_t>();
    for (int32_t i = 0; i < out_blob->shape().elem_cnt(); ++i) {
      *(dptr + i) = rand_bool_gen->GetNextBool() ? 1 : 0;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("coin_flip")
    .SetCreateFn<CoinFlipKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("out", 0) == DataType::kInt8));

namespace {

void ImageRandomCropImpl(const TensorBuffer* in_buffer, TensorBuffer* out_buffer,
                         RandomCropGenerator* random_crop_gen) {
  cv::Mat image = GenCvMat4ImageBuffer(*in_buffer);
  int W = image.cols;
  int H = image.rows;
  cv::Mat image_roi;
  CropWindow crop;
  random_crop_gen->GenerateCropWindow({H, W}, &crop);
  const int y = crop.anchor.At(0);
  const int x = crop.anchor.At(1);
  const int new_h = crop.shape.At(0);
  const int new_w = crop.shape.At(1);
  CHECK(new_w > 0 && new_w <= W);
  CHECK(new_h > 0 && new_h <= H);
  cv::Rect roi(x, y, new_w, new_h);
  image(roi).copyTo(image_roi);
  image = image_roi;
  W = image.cols;
  H = image.rows;

  CHECK(image.isContinuous());
  const int c = in_buffer->shape().At(2);
  CHECK_EQ(c, image.channels());
  Shape image_shape({H, W, c});
  out_buffer->Resize(image_shape, in_buffer->data_type());
  memcpy(out_buffer->mut_data<>(), image.ptr(), out_buffer->nbytes());
}

}  // namespace

class ImageRandomCropKernel final : public user_op::OpKernel {
 public:
  ImageRandomCropKernel() = default;
  ~ImageRandomCropKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateRandomCropKernelState(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* crop_window_generators = dynamic_cast<RandomCropKernelState*>(state);
    CHECK_NOTNULL(crop_window_generators);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = out_blob->shape().elem_cnt();
    CHECK(record_num > 0);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(out_blob->shape(), in_blob->shape());
    const TensorBuffer* in_buffers = in_blob->dptr<TensorBuffer>();
    TensorBuffer* out_buffers = out_blob->mut_dptr<TensorBuffer>();
    MultiThreadLoop(record_num, [&](size_t i) {
      ImageRandomCropImpl(in_buffers + i, out_buffers + i, crop_window_generators->GetGenerator(i));
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("image_random_crop")
    .SetCreateFn<ImageRandomCropKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
