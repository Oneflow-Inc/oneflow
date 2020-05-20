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

namespace {

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
    // int64_t in_image_elem_cnt = in_H * in_W * C;

    std::string output_layout = ctx->Attr<std::string>("output_layout");
    const ShapeView& out_shape = out_blob->shape();
    CHECK_EQ(output_layout, "NCHW");  // TODO(chengcheng): support NHWC
    CHECK_EQ(out_shape.At(0), N);
    CHECK_EQ(out_shape.At(1), C);
    int64_t out_H = out_shape.At(2);
    int64_t out_W = out_shape.At(3);
    // int64_t out_image_elem_cnt = out_H * out_W * C;
    CHECK(in_H == out_H && in_W == out_W);  // TODO(chengcheng): support crop
    int64_t H = in_H;
    int64_t W = in_W;
    int64_t one_sample_elem_cnt = C * H * W;

    MultiThreadLoop(record_num, [&](size_t i) {
      CMN1Sample(C, H, W, in_dptr + one_sample_elem_cnt * i, out_dptr + one_sample_elem_cnt * i,
                 mirror.at(i), mean_vec, inv_std_vec);
    });
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
