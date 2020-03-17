#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"

namespace oneflow {

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

class ResizeToStaticShapeKernel final : public user_op::OpKernel {
 public:
  ResizeToStaticShapeKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ResizeToStaticShapeKernel() = default;
  ~ResizeToStaticShapeKernel() = default;

 private:
  void ResizePartSamples(int32_t part_id, int32_t part_num, int64_t record_num,
                         TensorBuffer* buffers, int rsz_h, int rsz_w, int C, uint8_t* out_dptr,
                         const std::string& interp_type) {
    BalancedSplitter bs(record_num, part_num);
    Range range = bs.At(part_id);
    int64_t one_sample_elem_cnt = rsz_h * rsz_w * C;
    CHECK(C == 3 || C == 1);
    FOR_RANGE(int32_t, i, range.begin(), range.end()) {
      TensorBuffer* buffer = buffers + i;
      uint8_t* dptr = out_dptr + one_sample_elem_cnt * i;
      const Shape& in_shape = buffer->shape();
      CHECK(in_shape.NumAxes() == 3);  // {H, W, C}
      int H = in_shape.At(0);
      int W = in_shape.At(1);
      CHECK_EQ(C, in_shape.At(2));
      int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
      const cv::Mat image = CreateMatFromPtr(H, W, channel_flag, buffer->data<uint8_t>());
      cv::Mat rsz_image = CreateMatFromPtr(rsz_h, rsz_w, channel_flag, dptr);
      int opencv_inter_type = GetOpencvInterp(interp_type);
      cv::resize(image, rsz_image, cv::Size(rsz_w, rsz_h), 0, 0, opencv_inter_type);
    }
  }

  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = in_blob->shape().At(0);
    CHECK(record_num > 0);

    TensorBuffer* buffers = in_blob->mut_dptr<TensorBuffer>();
    uint8_t* out_dptr = out_blob->mut_dptr<uint8_t>();
    const ShapeView& out_shape = out_blob->shape();
    CHECK(out_shape.NumAxes() == 4);  // {N, H, W, C}
    int rsz_h = out_shape.At(1);
    int rsz_w = out_shape.At(2);
    int C = out_shape.At(3);
    std::string interp_type = ctx->GetAttr<std::string>("interp_type");

    ThreadPool* thread_pool = Global<ThreadMgr>::Get()->compute_thread_pool();
    int32_t thread_num = thread_pool->thread_num();
    int32_t part_num = std::min(static_cast<int32_t>(record_num), thread_num);
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      thread_pool->AddWork([&bc, part_id, part_num, record_num, buffers, rsz_h, rsz_w, C, out_dptr,
                            &interp_type, this]() {
        ResizePartSamples(part_id, part_num, record_num, buffers, rsz_h, rsz_w, C, out_dptr,
                          interp_type);
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  }
};

REGISTER_USER_KERNEL("Resize")
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

}  // namespace oneflow
