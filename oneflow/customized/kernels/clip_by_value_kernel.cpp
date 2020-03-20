#include "oneflow/customized/kernels/clip_by_value_kernel.h"

namespace oneflow {

template<typename T>
class ClipByScalarKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarKernel() = default;
  ~ClipByScalarKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    ClipByMinMaxFunctor<T> clip_fn(ctx->GetAttr<float>("min"), ctx->GetAttr<float>("max"));
    ClipUtil<T>::Forward(clip_fn, y->shape().elem_cnt(), x->dptr<T>(), y->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMinKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMinKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMinKernel() = default;
  ~ClipByScalarMinKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    ClipByMinFunctor<T> clip_fn(ctx->GetAttr<float>("min"));
    ClipUtil<T>::Forward(clip_fn, y->shape().elem_cnt(), x->dptr<T>(), y->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMaxKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMaxKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMaxKernel() = default;
  ~ClipByScalarMaxKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    ClipByMaxFunctor<T> clip_fn(ctx->GetAttr<float>("max"));
    ClipUtil<T>::Forward(clip_fn, y->shape().elem_cnt(), x->dptr<T>(), y->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarGradKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarGradKernel() = default;
  ~ClipByScalarGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    ClipByMinMaxGradFunctor<T> clip_fn(ctx->GetAttr<float>("min"), ctx->GetAttr<float>("max"));
    ClipUtil<T>::Backward(clip_fn, dx->shape().elem_cnt(), x->dptr<T>(), dy->dptr<T>(),
                          dx->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMinGradKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMinGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMinGradKernel() = default;
  ~ClipByScalarMinGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    ClipByMinGradFunctor<T> clip_fn(ctx->GetAttr<float>("min"));
    ClipUtil<T>::Backward(clip_fn, dx->shape().elem_cnt(), x->dptr<T>(), dy->dptr<T>(),
                          dx->mut_dptr<T>());
  }
};

template<typename T>
class ClipByScalarMaxGradKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  ClipByScalarMaxGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ClipByScalarMaxGradKernel() = default;
  ~ClipByScalarMaxGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    ClipByMaxGradFunctor<T> clip_fn(ctx->GetAttr<float>("max"));
    ClipUtil<T>::Backward(clip_fn, dx->shape().elem_cnt(), x->dptr<T>(), dy->dptr<T>(),
                          dx->mut_dptr<T>());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CLIP_KERNELS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
