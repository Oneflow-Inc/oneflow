#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

namespace {

template<typename dtype>
std::function<bool(const user_op::KernelRegContext& ctx)> MakeIsMatchedPred(
    DeviceType device_type) {
  return [device_type](const user_op::KernelRegContext& ctx) {
    const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);
    return ctx.device_type() == device_type && y_desc->data_type() == GetDataType<dtype>::value;
  };
}
}  // namespace

template<typename T>
class CpuAvgPool1DKernel final : public user_op::OpKernel {
 public:
  CpuAvgPool1DKernel() = default;
  ~CpuAvgPool1DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 1);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuAvgFWCompute(ctx, state);
  };
};

template<typename T>
class CpuAvgPool1DGradKernel final : public user_op::OpKernel {
 public:
  CpuAvgPool1DGradKernel() = default;
  ~CpuAvgPool1DGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 1);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuAvgBWCompute(ctx, state);
  };
};
template<typename T>
class CpuAvgPool2DKernel final : public user_op::OpKernel {
 public:
  CpuAvgPool2DKernel() = default;
  ~CpuAvgPool2DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 2);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuAvgFWCompute(ctx, state);
  };
};

template<typename T>
class CpuAvgPool2DGradKernel final : public user_op::OpKernel {
 public:
  CpuAvgPool2DGradKernel() = default;
  ~CpuAvgPool2DGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 2);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuAvgBWCompute(ctx, state);
  };
};
template<typename T>
class CpuAvgPool3DKernel final : public user_op::OpKernel {
 public:
  CpuAvgPool3DKernel() = default;
  ~CpuAvgPool3DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 3);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuAvgFWCompute(ctx, state);
  };
};

template<typename T>
class CpuAvgPool3DGradKernel final : public user_op::OpKernel {
 public:
  CpuAvgPool3DGradKernel() = default;
  ~CpuAvgPool3DGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 3);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuAvgBWCompute(ctx, state);
  };
};

template<typename T>
class CpuMaxPool1DKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool1DKernel() = default;
  ~CpuMaxPool1DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 1);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuMaxFWCompute(ctx, state);
  };
};

template<typename T>
class CpuMaxPool1DGradKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool1DGradKernel() = default;
  ~CpuMaxPool1DGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 1);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuMaxBWCompute(ctx, state);
  };
};
template<typename T>
class CpuMaxPool2DKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool2DKernel() = default;
  ~CpuMaxPool2DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 2);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuMaxFWCompute(ctx, state);
  };
};

template<typename T>
class CpuMaxPool2DGradKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool2DGradKernel() = default;
  ~CpuMaxPool2DGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 2);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuMaxBWCompute(ctx, state);
  };
};
template<typename T>
class CpuMaxPool3DKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool3DKernel() = default;
  ~CpuMaxPool3DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 3);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuMaxFWCompute(ctx, state);
  };
};

template<typename T>
class CpuMaxPool3DGradKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool3DGradKernel() = default;
  ~CpuMaxPool3DGradKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return PoolKernelUtil<T>::CreateOpKernelState(ctx, 3);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolKernelUtil<T>::CpuMaxBWCompute(ctx, state);
  };
};

#define REGISTER_POOL_CPU_KERNEL(dtype)                              \
  REGISTER_USER_KERNEL("avg_pool_1d")                                \
      .SetCreateFn<CpuAvgPool1DKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_1d_grad")                           \
      .SetCreateFn<CpuAvgPool1DGradKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_2d")                                \
      .SetCreateFn<CpuAvgPool2DKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                           \
      .SetCreateFn<CpuAvgPool2DGradKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_3d")                                \
      .SetCreateFn<CpuAvgPool3DKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_3d_grad")                           \
      .SetCreateFn<CpuAvgPool3DGradKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_1d")                                \
      .SetCreateFn<CpuMaxPool1DKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_1d_grad")                           \
      .SetCreateFn<CpuMaxPool1DGradKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_2d")                                \
      .SetCreateFn<CpuMaxPool2DKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                           \
      .SetCreateFn<CpuMaxPool2DGradKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_3d")                                \
      .SetCreateFn<CpuMaxPool3DKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                           \
      .SetCreateFn<CpuMaxPool3DGradKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU));

REGISTER_POOL_CPU_KERNEL(float)
REGISTER_POOL_CPU_KERNEL(double)

}  // namespace oneflow
