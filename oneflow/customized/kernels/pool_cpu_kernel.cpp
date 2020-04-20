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
class AvgPool1DCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool1DCpuKernel() = default;
  ~AvgPool1DCpuKernel() = default;

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
class AvgPool1DGradCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool1DGradCpuKernel() = default;
  ~AvgPool1DGradCpuKernel() = default;

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
class AvgPool2DCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2DCpuKernel() = default;
  ~AvgPool2DCpuKernel() = default;

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
class AvgPool2DGradCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool2DGradCpuKernel() = default;
  ~AvgPool2DGradCpuKernel() = default;

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
class AvgPool3DCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool3DCpuKernel() = default;
  ~AvgPool3DCpuKernel() = default;

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
class AvgPool3DGradCpuKernel final : public user_op::OpKernel {
 public:
  AvgPool3DGradCpuKernel() = default;
  ~AvgPool3DGradCpuKernel() = default;

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
class MaxPool1DCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool1DCpuKernel() = default;
  ~MaxPool1DCpuKernel() = default;

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
class MaxPool1DGradCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool1DGradCpuKernel() = default;
  ~MaxPool1DGradCpuKernel() = default;

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
class MaxPool2DCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2DCpuKernel() = default;
  ~MaxPool2DCpuKernel() = default;

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
class MaxPool2DGradCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool2DGradCpuKernel() = default;
  ~MaxPool2DGradCpuKernel() = default;

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
class MaxPool3DCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DCpuKernel() = default;
  ~MaxPool3DCpuKernel() = default;

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
class MaxPool3DGradCpuKernel final : public user_op::OpKernel {
 public:
  MaxPool3DGradCpuKernel() = default;
  ~MaxPool3DGradCpuKernel() = default;

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
      .SetCreateFn<AvgPool1DCpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_1d_grad")                           \
      .SetCreateFn<AvgPool1DGradCpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_2d")                                \
      .SetCreateFn<AvgPool2DCpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                           \
      .SetCreateFn<AvgPool2DGradCpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_3d")                                \
      .SetCreateFn<AvgPool3DCpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("avg_pool_3d_grad")                           \
      .SetCreateFn<AvgPool3DGradCpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_1d")                                \
      .SetCreateFn<MaxPool1DCpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_1d_grad")                           \
      .SetCreateFn<MaxPool1DGradCpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_2d")                                \
      .SetCreateFn<MaxPool2DCpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                           \
      .SetCreateFn<MaxPool2DGradCpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_3d")                                \
      .SetCreateFn<MaxPool3DCpuKernel<dtype>>()                      \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU)); \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                           \
      .SetCreateFn<MaxPool3DGradCpuKernel<dtype>>()                  \
      .SetIsMatchedPred(MakeIsMatchedPred<dtype>(DeviceType::kCPU));

REGISTER_POOL_CPU_KERNEL(float)
REGISTER_POOL_CPU_KERNEL(double)

}  // namespace oneflow
