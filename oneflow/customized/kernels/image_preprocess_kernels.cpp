#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"

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
  ResizeToStaticShapeKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ResizeToStaticShapeKernel() = delete;
  ~ResizeToStaticShapeKernel() override = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
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
    std::string interp_type = ctx->GetAttr<std::string>("interp_type");
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
};

REGISTER_USER_KERNEL("ImageResize")
    .SetCreateFn([](user_op::KernelInitContext* ctx) { return new ResizeToStaticShapeKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kTensorBuffer
          && out_tensor->data_type() == DataType::kUInt8) {
        return true;
      }
      return false;
    });

class CropMirrorNormalizeFromStaticShapeToFloatKernel final : public user_op::OpKernel {
 public:
  CropMirrorNormalizeFromStaticShapeToFloatKernel(user_op::KernelInitContext* ctx)
      : user_op::OpKernel(ctx) {
    mean_vec_ = ctx->GetAttr<std::vector<float>>("mean");
    inv_std_vec_ = ctx->GetAttr<std::vector<float>>("std");
    std::string color_space = ctx->GetAttr<std::string>("color_space");
    int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
    CHECK(mean_vec_.size() == 1 || mean_vec_.size() == C);
    CHECK(inv_std_vec_.size() == 1 || inv_std_vec_.size() == C);

    for (auto& elem : inv_std_vec_) { elem = 1.0f / elem; }

    if (mean_vec_.size() == 1) { mean_vec_.resize(C, mean_vec_.at(0)); }
    if (inv_std_vec_.size() == 1) { inv_std_vec_.resize(C, inv_std_vec_.at(0)); }
  }
  CropMirrorNormalizeFromStaticShapeToFloatKernel() = delete;
  ~CropMirrorNormalizeFromStaticShapeToFloatKernel() override = default;

 private:
  void CMN1Sample(int64_t C, int64_t H, int64_t W, const uint8_t* in_dptr, float* out_dptr,
                  int8_t mirror) {
    if (mirror) {
      for (int64_t c = 0; c < C; ++c) {
        float mean = mean_vec_.at(c);
        float inv_std = inv_std_vec_.at(c);
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = 0; w < W; ++w) {
            int64_t mirror_w = W - 1 - w;
            int64_t in_offset = h * W * C + mirror_w * C + c;  // N, H, W, C
            int64_t out_offset = c * H * W + h * W + w;        // N, C, H, W
            out_dptr[out_offset] = (static_cast<float>(in_dptr[in_offset]) - mean) * inv_std;
          }
        }
      }
    } else {
      for (int64_t c = 0; c < C; ++c) {
        float mean = mean_vec_.at(c);
        float inv_std = inv_std_vec_.at(c);
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

  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* mirror_blob = ctx->Tensor4ArgNameAndIndex("mirror", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = in_blob->shape().At(0);
    CHECK(record_num > 0);
    if (mirror_blob) {
      CHECK_EQ(record_num, mirror_blob->shape().elem_cnt());
      mirror_.resize(record_num);
      for (int32_t i = 0; i < record_num; ++i) {
        mirror_.at(i) = *(mirror_blob->mut_dptr<int8_t>() + i);
      }
    } else {
      mirror_.resize(record_num, 0);
    }

    const uint8_t* in_dptr = in_blob->dptr<uint8_t>();
    float* out_dptr = out_blob->mut_dptr<float>();

    const ShapeView& in_shape = in_blob->shape();
    int64_t N = in_shape.At(0);
    int64_t in_H = in_shape.At(1);
    int64_t in_W = in_shape.At(2);
    int64_t C = in_shape.At(3);
    // int64_t in_image_elem_cnt = in_H * in_W * C;

    std::string output_layout = ctx->GetAttr<std::string>("output_layout");
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
                 mirror_.at(i));
    });
  }

  std::vector<float> mean_vec_;
  std::vector<float> inv_std_vec_;
  std::vector<int8_t> mirror_;
};

REGISTER_USER_KERNEL("CropMirrorNormalize")
    .SetCreateFn([](user_op::KernelInitContext* ctx) {
      return new CropMirrorNormalizeFromStaticShapeToFloatKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kUInt8
          && out_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

class CoinFlipKernel final : public user_op::OpKernel {
 public:
  CoinFlipKernel(user_op::KernelInitContext* ctx)
      : user_op::OpKernel(ctx),
        dis_(ctx->GetAttr<float>("probability")),
        rng_(ctx->GetAttr<int64_t>("seed") == -1 ? NewRandomSeed()
                                                 : ctx->GetAttr<int64_t>("seed")) {
    /*
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    if (seed == -1) { seed = NewRandomSeed(); }
    rng_ = std::mt19937(seed);
    */
  }
  CoinFlipKernel() = delete;
  ~CoinFlipKernel() override = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int8_t* dptr = out_blob->mut_dptr<int8_t>();
    for (int32_t i = 0; i < out_blob->shape().elem_cnt(); ++i) { *(dptr + i) = dis_(rng_) ? 1 : 0; }
  }

  std::bernoulli_distribution dis_;
  std::mt19937 rng_;
};

REGISTER_USER_KERNEL("CoinFlip")
    .SetCreateFn([](user_op::KernelInitContext* ctx) { return new CoinFlipKernel(ctx); })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && out_tensor->data_type() == DataType::kInt8) {
        return true;
      }
      return false;
    });

}  // namespace oneflow
