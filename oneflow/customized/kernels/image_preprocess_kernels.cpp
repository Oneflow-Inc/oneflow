#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"
#include "oneflow/customized/kernels/random_seed_util.h"

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

    TensorBuffer* buffers = in_blob->mut_dptr<TensorBuffer>();
    uint8_t* out_dptr = out_blob->mut_dptr<uint8_t>();
    const ShapeView& out_shape = out_blob->shape();
    CHECK(out_shape.NumAxes() == 4);  // {N, H, W, C}
    int rsz_h = out_shape.At(1);
    int rsz_w = out_shape.At(2);
    int C = out_shape.At(3);
    CHECK(C == 3 || C == 1);
    int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
    std::string interp_type = ctx->Attr<std::string>("interp_type");
    int64_t one_sample_elem_cnt = rsz_h * rsz_w * C;
    int opencv_inter_type = GetOpencvInterp(interp_type);

    MultiThreadLoop(record_num, [&](size_t i) {
      TensorBuffer* buffer = buffers + i;
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
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kTensorBuffer
          && out_tensor->data_type() == DataType::kUInt8) {
        return true;
      }
      return false;
    });

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

    TensorBuffer* in_buffers = in_blob->mut_dptr<TensorBuffer>();
    TensorBuffer* out_buffers = out_blob->mut_dptr<TensorBuffer>();
    int64_t resize_shorter = ctx->Attr<int64_t>("resize_shorter");

    MultiThreadLoop(record_num, [&](size_t i) {
      TensorBuffer* in_buffer = in_buffers + i;
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
      std::string interp_type = ctx->Attr<std::string>("interp_type");
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
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kTensorBuffer
          && out_tensor->data_type() == DataType::kTensorBuffer) {
        return true;
      }
      return false;
    });

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
/*
void CMN1Sample(int64_t C, int64_t H, int64_t W, const uint8_t* in_dptr, float* out_dptr,
                int8_t mirror, const std::vector<float>& mean_vec,
                const std::vector<float>& inv_std_vec) {
  if (mirror) {
    for (int64_t c = 0; c < C; ++c) {
      float mean = mean_vec.at(c);
      float inv_std = inv_std_vec.at(c);
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          int64_t mirrorw = W - 1 - w;
          int64_t in_offset = h * W * C + mirrorw * C + c;  // N, H, W, C
          int64_t out_offset = c * H * W + h * W + w;       // N, C, H, W
          out_dptr[out_offset] = (static_cast<float>(in_dptr[in_offset]) - mean) * inv_std;
        }
      }
    }
  } else {
    for (int64_t c = 0; c < C; ++c) {
      float mean = mean_vec.at(c);
      float inv_std = inv_std_vec.at(c);
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          int64_t in_offset = h * W * C + w * C + c;   // N, H, W, C
          int64_t out_offset = c * H * W + h * W + w;  // N, C, H, W
          out_dptr[out_offset] = (static_cast<float>(in_dptr[in_offset]) - mean) * inv_std;
        }
      }
    }
  }
}
*/

}  // namespace

class CropMirrorNormalizeFromStaticShapeToFloatKernel final : public user_op::OpKernel {
 public:
  CropMirrorNormalizeFromStaticShapeToFloatKernel() = default;
  ~CropMirrorNormalizeFromStaticShapeToFloatKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<float> mean_vec = ctx->Attr<std::vector<float>>("mean");
    std::vector<float> inv_std_vec = ctx->Attr<std::vector<float>>("std");
    std::vector<int8_t> mirror;
    std::string color_space = ctx->Attr<std::string>("color_space");
    int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
    CHECK(mean_vec.size() == 1 || mean_vec.size() == C);
    CHECK(inv_std_vec.size() == 1 || inv_std_vec.size() == C);

    for (auto& elem : inv_std_vec) { elem = 1.0f / elem; }

    if (mean_vec.size() == 1) { mean_vec.resize(C, mean_vec.at(0)); }
    if (inv_std_vec.size() == 1) { inv_std_vec.resize(C, inv_std_vec.at(0)); }

    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* mirrorblob = ctx->Tensor4ArgNameAndIndex("mirror", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = in_blob->shape().At(0);
    CHECK(record_num > 0);
    if (mirrorblob) {
      CHECK_EQ(record_num, mirrorblob->shape().elem_cnt());
      mirror.resize(record_num);
      for (int32_t i = 0; i < record_num; ++i) {
        mirror.at(i) = *(mirrorblob->mut_dptr<int8_t>() + i);
      }
    } else {
      mirror.resize(record_num, 0);
    }

    const uint8_t* in_dptr = in_blob->dptr<uint8_t>();
    float* out_dptr = out_blob->mut_dptr<float>();

    const ShapeView& in_shape = in_blob->shape();
    int64_t N = in_shape.At(0);
    int64_t in_H = in_shape.At(1);
    int64_t in_W = in_shape.At(2);
    CHECK_EQ(C, in_shape.At(3));
    int64_t in_image_elem_cnt = in_H * in_W * C;

    std::string output_layout = ctx->Attr<std::string>("output_layout");
    const ShapeView& out_shape = out_blob->shape();
    CHECK_EQ(out_shape.NumAxes(), 4);
    CHECK_EQ(out_shape.At(0), N);
    float crop_pos_y = ctx->Attr<float>("crop_pos_y");
    float crop_pos_x = ctx->Attr<float>("crop_pos_x");
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

REGISTER_USER_KERNEL("crop_mirror_normalize")
    .SetCreateFn<CropMirrorNormalizeFromStaticShapeToFloatKernel>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kUInt8
          && out_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

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

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    float prob = ctx->Attr<float>("probability");
    int64_t seed = GetOpKernelRandomSeed(ctx);
    std::shared_ptr<RandBoolGen> rand_bool_gen(new RandBoolGen(prob, seed));
    return rand_bool_gen;
  }

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
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && out_tensor->data_type() == DataType::kInt8) {
        return true;
      }
      return false;
    });

}  // namespace oneflow
