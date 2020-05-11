#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

template<typename T>
void ConvertImageCvMat(const cv::Mat& input, cv::Mat* output);

template<>
void ConvertImageCvMat<uint8_t>(const cv::Mat& input, cv::Mat* output) {
  input.convertTo(*output, CV_8U);
}

template<>
void ConvertImageCvMat<float>(const cv::Mat& input, cv::Mat* output) {
  input.convertTo(*output, CV_32F);
}

template<typename T>
void CopyCvMatToImageTensor(const cv::Mat& image_mat, user_op::Tensor* image_tensor,
                            int image_idx) {
  CHECK_EQ(image_mat.rows, image_tensor->shape().At(1));
  CHECK_EQ(image_mat.cols, image_tensor->shape().At(2));
  CHECK_EQ(image_mat.channels(), image_tensor->shape().At(3));

  int rows = image_mat.rows;
  int cols = image_mat.cols * image_mat.channels();
  if (image_mat.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  T* image_ptr = image_tensor->mut_dptr<T>() + image_idx * rows * cols;
  FOR_RANGE(int64_t, i, 0, rows) {
    CopyElem(image_mat.ptr<T>(i), image_ptr, cols);
    image_ptr += cols;
  }
}

template<typename T>
struct ImageBatchAlignIsMatchedPred {
  static bool Impl(const user_op::KernelRegContext& ctx) {
    const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
    const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
    return ctx.device_type() == DeviceType::kCPU
           && in_tensor->data_type() == DataType::kTensorBuffer
           && out_tensor->data_type() == GetDataType<T>::value;
  }
};

}  // namespace

template<typename T>
class ImageBatchAlignKernel final : public user_op::OpKernel {
 public:
  ImageBatchAlignKernel() = default;
  ~ImageBatchAlignKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_GT(in_tensor->shape().elem_cnt(), 0);
    CHECK_EQ(in_tensor->shape().elem_cnt(), out_tensor->shape().At(0));
    int batch_height = out_tensor->shape().At(1);
    int batch_width = out_tensor->shape().At(2);

    MultiThreadLoop(out_tensor->shape().At(0), [&](size_t i) {
      const TensorBuffer& origin_image_buffer = in_tensor->dptr<TensorBuffer>()[i];
      const cv::Mat origin_image_mat = GenCvMat4ImageBuffer(origin_image_buffer);
      cv::Mat dst = cv::Mat::zeros(cv::Size(batch_width, batch_height), origin_image_mat.type());
      origin_image_mat.copyTo(dst(cv::Rect(0, 0, origin_image_mat.cols, origin_image_mat.rows)));
      ConvertImageCvMat<T>(dst, &dst);
      CopyCvMatToImageTensor<T>(dst, out_tensor, i);
    });
  }

  void InferShape(user_op::KernelInferContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int num_images = in_tensor->shape().elem_cnt();
    int64_t max_height = 0;
    int64_t max_width = 0;
    int64_t channels = out_tensor->shape().At(3);
    FOR_RANGE(int, i, 0, num_images) {
      const TensorBuffer& image_buffer = in_tensor->dptr<TensorBuffer>()[i];
      max_height = std::max(max_height, image_buffer.shape().At(0));
      max_width = std::max(max_width, image_buffer.shape().At(1));
      CHECK_EQ(image_buffer.shape().At(2), channels);
    }
    int32_t alignment = ctx->GetAttr<int32_t>("alignment");
    max_height = RoundUp(max_height, alignment);
    max_width = RoundUp(max_width, alignment);
    auto* mut_shape_view = out_tensor->mut_shape();
    mut_shape_view->Set(0, num_images);
    mut_shape_view->Set(1, max_height);
    mut_shape_view->Set(2, max_width);
    // TODO: need to check static shape can hold dynamic shape
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_IMAGE_BATCH_ALIGN_KERNEL(dtype)   \
  REGISTER_USER_KERNEL("image_batch_align")        \
      .SetCreateFn<ImageBatchAlignKernel<dtype>>() \
      .SetIsMatchedPred(ImageBatchAlignIsMatchedPred<dtype>::Impl);

REGISTER_IMAGE_BATCH_ALIGN_KERNEL(uint8_t)
REGISTER_IMAGE_BATCH_ALIGN_KERNEL(float)

}  // namespace oneflow
