#include "oneflow/customized/kernels/clip_by_value_kernel.h"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename F>
__global__ void CudaClipForward(F clip_fn, int64_t num_values, const T* x, T* y) {
  ClipUtil<T>::Forward(clip_fn, num_values, x, y);
}

template<typename T, typename F>
__global__ void CudaClipBackward(F clip_fn, int64_t num_values, const T* x, const T* dy, T* dx) {
  ClipUtil<T>::Backward(clip_fn, num_values, x, dy, dx);
}

}  // namespace

template<typename T>
class ClipByScalarKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarKernel() = default;
  ~ClipByScalarKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    int64_t elem_cnt = y->shape().elem_cnt();
    ClipByMinMaxFunctor<T> clip_fn(ctx->GetAttr<float>("min"), ctx->GetAttr<float>("max"));
    CudaClipForward<<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                      ctx->device_ctx()->cuda_stream()>>>(clip_fn, elem_cnt, x->dptr<T>(),
                                                          y->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMinKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMinKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMinKernel() = default;
  ~ClipByScalarMinKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    int64_t elem_cnt = y->shape().elem_cnt();
    ClipByMinFunctor<T> clip_fn(ctx->GetAttr<float>("min"));
    CudaClipForward<<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                      ctx->device_ctx()->cuda_stream()>>>(clip_fn, elem_cnt, x->dptr<T>(),
                                                          y->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMaxKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMaxKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMaxKernel() = default;
  ~ClipByScalarMaxKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    int64_t elem_cnt = y->shape().elem_cnt();
    ClipByMaxFunctor<T> clip_fn(ctx->GetAttr<float>("max"));
    CudaClipForward<<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                      ctx->device_ctx()->cuda_stream()>>>(clip_fn, elem_cnt, x->dptr<T>(),
                                                          y->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarGradKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarGradKernel() = default;
  ~ClipByScalarGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int64_t elem_cnt = dx->shape().elem_cnt();
    ClipByMinMaxGradFunctor<T> clip_fn(ctx->GetAttr<float>("min"), ctx->GetAttr<float>("max"));
    CudaClipBackward<<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                       ctx->device_ctx()->cuda_stream()>>>(clip_fn, elem_cnt, x->dptr<T>(),
                                                           dy->dptr<T>(), dx->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMinGradKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMinGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMinGradKernel() = default;
  ~ClipByScalarMinGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int64_t elem_cnt = dx->shape().elem_cnt();
    ClipByMinGradFunctor<T> clip_fn(ctx->GetAttr<float>("min"));
    CudaClipBackward<<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                       ctx->device_ctx()->cuda_stream()>>>(clip_fn, elem_cnt, x->dptr<T>(),
                                                           dy->dptr<T>(), dx->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMaxGradKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMaxGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMaxGradKernel() = default;
  ~ClipByScalarMaxGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    int64_t elem_cnt = dx->shape().elem_cnt();
    ClipByMaxGradFunctor<T> clip_fn(ctx->GetAttr<float>("max"));
    CudaClipBackward<<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                       ctx->device_ctx()->cuda_stream()>>>(clip_fn, elem_cnt, x->dptr<T>(),
                                                           dy->dptr<T>(), dx->mut_dptr<T>());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CLIP_KERNELS, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
