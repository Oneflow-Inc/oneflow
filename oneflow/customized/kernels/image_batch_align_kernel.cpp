#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

template<typename T, typename F>
void CopyFromTensorBuffer(T* image_ptr, const TensorBuffer& image_buffer, const int batch_height,
                          const int batch_width, const int channels) {
  CHECK_EQ(image_buffer.shape().NumAxes(), 3);
  const int h = image_buffer.shape().At(0);
  const int w = image_buffer.shape().At(1);
  const int c = image_buffer.shape().At(2);
  CHECK_LE(h, batch_height);
  CHECK_LE(w, batch_width);
  CHECK_EQ(c, channels);
  FOR_RANGE(int, i, 0, h) {
    const F* from = image_buffer.data<F>() + i * w * c;
    T* to = image_ptr + i * batch_width * channels;
    CopyElem(from, to, w * c);
  }
}

template<typename T>
struct ImageCopier final {
#define MAKE_COPY_FROM_TENSOR_BUFFER_SWITCH_ENTRY(func_name, F) func_name<T, F>
  DEFINE_STATIC_SWITCH_FUNC(void, CopyFromTensorBuffer, MAKE_COPY_FROM_TENSOR_BUFFER_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(IMAGE_DATA_TYPE_SEQ))
#undef MAKE_COPY_FROM_TENSOR_BUFFER_SWITCH_ENTRY
};

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
    CHECK_EQ(out_tensor->shape().NumAxes(), 4);
    CHECK_EQ(out_tensor->shape().At(0), in_tensor->shape().elem_cnt());
    int batch_height = out_tensor->shape().At(1);
    int batch_width = out_tensor->shape().At(2);
    int channels = out_tensor->shape().At(3);
    int num_elems = batch_height * batch_width * channels;
    memset(out_tensor->mut_dptr(), 0,
           out_tensor->shape().elem_cnt() * GetSizeOfDataType(out_tensor->data_type()));
    MultiThreadLoop(out_tensor->shape().At(0), [&](size_t i) {
      const TensorBuffer& origin_image_buffer = in_tensor->dptr<TensorBuffer>()[i];
      T* out_ptr = out_tensor->mut_dptr<T>() + i * num_elems;
      ImageCopier<T>::SwitchCopyFromTensorBuffer(SwitchCase(origin_image_buffer.data_type()),
                                                 out_ptr, origin_image_buffer, batch_height,
                                                 batch_width, channels);
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
    int32_t alignment = ctx->Attr<int32_t>("alignment");
    max_height = RoundUp(max_height, alignment);
    max_width = RoundUp(max_width, alignment);
    auto* mut_shape_view = out_tensor->mut_shape();
    mut_shape_view->Set(0, num_images);
    mut_shape_view->Set(1, max_height);
    mut_shape_view->Set(2, max_width);
    // TODO(wenxiao): need to check static shape can hold dynamic shape
    // Tensor should add Capacity method
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
