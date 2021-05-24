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
#include "oneflow/core/common/data_type_converter.h"
#include "oneflow/user/image/image_util.h"
#include "oneflow/user/kernels/random_crop_kernel_state.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

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

namespace {

template<typename F, typename T>
void ImageHsvStatic(const TensorBuffer& input_buffer, TensorBuffer* output_buffer,
                    std::vector<float>& mat) {
  // TODO
  CHECK_EQ(input_buffer.shape().NumAxes(), 3);
  int h = input_buffer.shape().At(0);
  int w = input_buffer.shape().At(1);
  int c = input_buffer.shape().At(2);
  CHECK_EQ(c, 3);
  FOR_RANGE(int, i, 0, h) {
    auto* row_ptr = input_buffer.data<F>() + i * w * c;
    FOR_RANGE(int, j, 0, w) {
      std::vector<float> v_in;
      FOR_RANGE(int, k, 0, c) { v_in.push_back(row_ptr[j * c + k]); }
      std::vector<float> v_out(c, 0);
      FOR_RANGE(int, v_i, 0, c) {
        FOR_RANGE(int, v_j, 0, c) { v_out.at(v_i) += mat.at(v_i * c + v_j) * v_in.at(v_j); }
      }
      FOR_RANGE(int, k, 0, c) {
        T adjust_value = ConvertSat<T>(v_out.at(k));
        memcpy(output_buffer->mut_data<T>() + i * w * c + j * c + k, &adjust_value, sizeof(T));
      }
    }
  }
}

template<typename F>
void ImageHsvImpl(const TensorBuffer& input_buffer, TensorBuffer* output_buffer,
                  std::vector<float>& mat, DataType outbuffer_dtype) {
  outbuffer_dtype =
      (outbuffer_dtype != DataType::kInvalidDataType) ? outbuffer_dtype : input_buffer.data_type();
  output_buffer->Resize(input_buffer.shape(), outbuffer_dtype);

  switch (outbuffer_dtype) {
    case DataType::kChar:
    case DataType::kInt8:
    case DataType::kUInt8:
      ImageHsvStatic<F, DataTypeToType<DataType::kUInt8>>(input_buffer, output_buffer, mat);
      break;
    case DataType::kFloat16:
    case DataType::kFloat:
      ImageHsvStatic<F, DataTypeToType<DataType::kFloat>>(input_buffer, output_buffer, mat);
    case DataType::kInt32:
    case DataType::kDouble:
    case DataType::kInt64:
    default: { LOG(FATAL) << "Invalid data type " << outbuffer_dtype; }
  }
}

#define MAKE_IMAGE_HSV_FROM_TENSOR_BUFFER_SWITCH_ENTRY(func_name, F) func_name<F>
DEFINE_STATIC_SWITCH_FUNC(void, ImageHsvImpl, MAKE_IMAGE_HSV_FROM_TENSOR_BUFFER_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(IMAGE_DATA_TYPE_SEQ));

#undef MAKE_IMAGE_HSV_FROM_TENSOR_BUFFER_SWITCH_ENTRY

}  // namespace

struct ImageHsvOpKernelState final : public user_op::OpKernelState {
  std::vector<float> rgb2yiq;
  std::vector<float> yiq2rgb;
  std::vector<float> hue_mat;
  std::vector<float> sat_mat;
  std::vector<float> val_mat;
  std::vector<float> final_mat;
};

std::shared_ptr<user_op::OpKernelState> CreateImageHsvOpKernelState(
    user_op::KernelInitContext* ctx, const std::string& hue_name,
    const std::string& saturation_name, const std::string& value_name) {
  std::shared_ptr<ImageHsvOpKernelState> state(new ImageHsvOpKernelState());

  state->rgb2yiq = {.299f, .587f, .114f, .596f, -.274f, -.321f, .211f, -.523f, .311f};
  state->yiq2rgb = {1, .956f, .621f, 1, -.272f, -.647f, 1, -1.107f, 1.705f};

  state->hue_mat = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  state->sat_mat = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  state->val_mat = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  state->final_mat.assign(state->yiq2rgb.begin(), state->yiq2rgb.end());

  auto SetHueMat = [](float hue, std::vector<float>& hue_mat) {
    const double PI = std::atan(1.0) * 4.0;
    const float h_rad = hue * PI / 180;
    hue_mat.at(4) = std::cos(h_rad);
    hue_mat.at(8) = std::cos(h_rad);
    hue_mat.at(5) = std::sin(h_rad);
    hue_mat.at(7) = -std::sin(h_rad);
  };

  auto SetSatMat = [](float saturation, std::vector<float>& sat_mat) {
    sat_mat.at(4) = saturation;
    sat_mat.at(8) = saturation;
  };

  auto SetValMat = [](float value, std::vector<float>& val_mat) {
    val_mat.at(0) = value;
    val_mat.at(4) = value;
    val_mat.at(8) = value;
  };

  auto Matmul = [](std::vector<float>& left_mat, std::vector<float>& right_mat) {
    CHECK_EQ(left_mat.size(), 9);
    CHECK_EQ(right_mat.size(), 9);
    std::vector<float> mid_vec(9, 0);
    FOR_RANGE(int32_t, m, 0, 3) {
      FOR_RANGE(int32_t, k, 0, 3) {
        FOR_RANGE(int32_t, n, 0, 3) {
          mid_vec.at(m * 3 + n) += left_mat.at(m * 3 + k) * right_mat.at(k * 3 + n);
        }
      }
    }
    left_mat.assign(mid_vec.begin(), mid_vec.end());
  };

  SetHueMat(ctx->Attr<float>(hue_name), state->hue_mat);
  SetSatMat(ctx->Attr<float>(saturation_name), state->sat_mat);
  SetValMat(ctx->Attr<float>(value_name), state->val_mat);
  Matmul(state->final_mat, state->hue_mat);
  Matmul(state->final_mat, state->sat_mat);
  Matmul(state->final_mat, state->val_mat);
  Matmul(state->final_mat, state->rgb2yiq);

  return std::move(state);
}

class ImageHsvKernel final : public user_op::OpKernel {
 public:
  ImageHsvKernel() = default;
  ~ImageHsvKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    return CreateImageHsvOpKernelState(ctx, "hue", "saturation", "value");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* hsv_state = dynamic_cast<ImageHsvOpKernelState*>(state);
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in_tensor->shape().NumAxes(), 1);
    CHECK_EQ(out_tensor->shape().NumAxes(), 1);
    const int64_t num_images = in_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);
    DataType outbuffer_dtype = ctx->Attr<DataType>("data_type");

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& in_buffer = in_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(in_buffer.shape().NumAxes(), 3);
      TensorBuffer* out_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      SwitchImageHsvImpl(SwitchCase(in_buffer.data_type()), in_buffer, out_buffer,
                         hsv_state->final_mat, outbuffer_dtype);
    });
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("image_hsv")
    .SetCreateFn<ImageHsvKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
