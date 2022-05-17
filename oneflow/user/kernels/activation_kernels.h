/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef _ONEFLOW_USER_KERNELS_ACTIVATION_KERNELS_H_
#define _ONEFLOW_USER_KERNELS_ACTIVATION_KERNELS_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct LeakyReluFunctor {
  OF_DEVICE_FUNC explicit LeakyReluFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const { return (x > 0) ? x : alpha * x; }
  const T alpha;
};

template<typename T>
struct LeakyReluGradFunctor {
  OF_DEVICE_FUNC explicit LeakyReluGradFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const { return (x > 0) ? dy : dy * alpha; }
  const T alpha;
};

template<typename T>
struct EluFunctor {
  OF_DEVICE_FUNC explicit EluFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0)) ? x : static_cast<T>(alpha * (exp(x) - static_cast<T>(1)));
  }
  const T alpha;
};

template<typename T>
struct EluGradFunctor {
  OF_DEVICE_FUNC explicit EluGradFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? dy : static_cast<T>(dy * alpha * (exp(x)));
  }
  const T alpha;
};

template<typename T>
struct CeluFunctor {
  OF_DEVICE_FUNC explicit CeluFunctor(float alpha) : alpha(alpha), inv_alpha(1.0f / alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0))
               ? x
               : static_cast<T>(alpha * (exp(x * inv_alpha) - static_cast<T>(1)));
  }
  const T alpha;
  const T inv_alpha;
};

template<typename T>
struct CeluGradFunctor {
  OF_DEVICE_FUNC explicit CeluGradFunctor(float alpha) : inv_alpha(1.0f / alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? dy : dy * static_cast<T>(exp(x * inv_alpha));
  }
  const T inv_alpha;
};

template<typename T>
struct HardswishFunctor {
  OF_DEVICE_FUNC T operator()(const T x) const {
    if (x <= static_cast<T>(-3)) {
      return static_cast<T>(0);
    } else if (x >= static_cast<T>(3)) {
      return x;
    } else {
      return (x * (x + static_cast<T>(3))) / static_cast<T>(6);
    }
  }
};

template<typename T>
struct HardswishGradFunctor {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const {
    if (x <= static_cast<T>(-3)) {
      return static_cast<T>(0);
    } else if (x >= static_cast<T>(3)) {
      return dy;
    } else {
      return ((x / static_cast<T>(3)) + static_cast<T>(0.5)) * dy;
    }
  }
};

template<typename T>
struct HardsigmoidFunctor {
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x <= static_cast<T>(-3))
      return static_cast<T>(0);
    else if (x >= static_cast<T>(3))
      return static_cast<T>(1);
    else
      return (x / static_cast<T>(6)) + static_cast<T>(0.5);
  }
};

template<typename T>
struct HardsigmoidGradFunctor {
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(-3) && x < static_cast<T>(3)) ? dy / static_cast<T>(6)
                                                             : static_cast<T>(0);
  }
};

template<typename T>
struct HardShrinkFunctor {
  OF_DEVICE_FUNC explicit HardShrinkFunctor(double lambd) : lambd(lambd) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x <= lambd && x >= -lambd) ? static_cast<T>(0) : x;
  }

  const T lambd;
};

template<typename T>
struct HardShrinkGradFunctor {
  OF_DEVICE_FUNC explicit HardShrinkGradFunctor(double lambd) : lambd(lambd) {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const {
    return y == static_cast<T>(0) ? static_cast<T>(0) : dy;
  }

  const T lambd;
};

template<typename T>
struct HardtanhFunctor {
  OF_DEVICE_FUNC explicit HardtanhFunctor(float min_val, float max_val)
      : min_val(min_val), max_val(max_val) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x <= min_val) {
      return min_val;
    } else if (x >= max_val) {
      return max_val;
    } else {
      return x;
    }
  }
  const T min_val;
  const T max_val;
};

template<typename T>
struct HardtanhGradFunctor {
  OF_DEVICE_FUNC explicit HardtanhGradFunctor(float min_val, float max_val)
      : min_val(min_val), max_val(max_val) {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const {
    return (y != min_val && y != max_val) ? dy : static_cast<T>(0);
  }
  const T min_val;
  const T max_val;
};

template<typename T>
struct MishFunctor {
  OF_DEVICE_FUNC explicit MishFunctor() {}
  OF_DEVICE_FUNC T operator()(T x) const {
    T soft_plus_val = log(static_cast<T>(1) + exp(x));
    T exp_val = exp(soft_plus_val);
    T neg_exp_val = exp(-soft_plus_val);
    T tanh_val = (exp_val - neg_exp_val) / (exp_val + neg_exp_val);
    return x * tanh_val;
  }
};

template<typename T>
struct MishGradFunctor {
  OF_DEVICE_FUNC explicit MishGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    T sp = log(static_cast<T>(1) + exp(x));
    T grad_sp = static_cast<T>(1) - exp(-sp);
    T tsp = (exp(sp) - exp(-sp)) / (exp(sp) + exp(-sp));
    T grad_tsp = (static_cast<T>(1) - tsp * tsp) * grad_sp;
    return dy * (x * grad_tsp + tsp);
  }
};

template<typename T>
struct SiluFunctor {
  OF_DEVICE_FUNC explicit SiluFunctor() {}
  OF_DEVICE_FUNC T operator()(T x) const { return (x / (static_cast<T>(1) + exp(-x))); }
};

template<typename T>
struct SiluGradFunctor {
  OF_DEVICE_FUNC explicit SiluGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    T sig = static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
    return dy * (sig * (static_cast<T>(1) + x * (static_cast<T>(1) - sig)));
  }
};

template<typename T>
struct SeluFunctor {
  OF_DEVICE_FUNC explicit SeluFunctor() {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0)) ? scale * x : scale * alpha * (exp(x) - static_cast<T>(1));
  }
  const T scale = 1.0507009873554804934193349852946;
  const T alpha = 1.6732632423543772848170429916717;
};

template<typename T>
struct SeluGradFunctor {
  OF_DEVICE_FUNC explicit SeluGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? scale * dy : dy * scale * alpha * (exp(x));
  }
  const T scale = 1.0507009873554804934193349852946;
  const T alpha = 1.6732632423543772848170429916717;
};

template<typename T>
struct SoftSignFunctor {
  OF_DEVICE_FUNC explicit SoftSignFunctor() {}
  OF_DEVICE_FUNC T operator()(T x) const { return x / (static_cast<T>(1) + abs(x)); }
};

template<typename T>
struct SoftSignGradFunctor {
  OF_DEVICE_FUNC explicit SoftSignGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    T val = (static_cast<T>(1) + abs(x));
    return dy / (val * val);
  }
};

template<typename T>
struct ThresholdFunctor {
  OF_DEVICE_FUNC explicit ThresholdFunctor(double threshold, double value)
      : threshold(threshold), value(value) {}
  OF_DEVICE_FUNC T operator()(T x) const { return (x > threshold) ? x : value; }
  const T threshold;
  const T value;
};

template<typename T>
struct ThresholdGradFunctor {
  OF_DEVICE_FUNC explicit ThresholdGradFunctor(double threshold) : threshold(threshold) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const { return (x > threshold) ? dy : static_cast<T>(0); }
  const T threshold;
};

template<typename T>
struct SoftplusFunctor {
  OF_DEVICE_FUNC explicit SoftplusFunctor(double beta, double threshold)
      : beta(beta), threshold(threshold) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x * beta) > threshold ? x : log(static_cast<T>(1.0) + exp(x * beta)) / beta;
  }

  const T beta;
  const T threshold;
};

template<typename T>
struct SoftplusGradFunctor {
  OF_DEVICE_FUNC explicit SoftplusGradFunctor(double beta, double threshold)
      : beta(beta), threshold(threshold) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    T z = exp(x * beta);
    return (x * beta) > threshold ? dy : dy * z / (z + static_cast<T>(1.0));
  }

  const T beta;
  const T threshold;
};

template<typename T>
struct ReluFunctor {
  OF_DEVICE_FUNC explicit ReluFunctor() {}
  OF_DEVICE_FUNC T operator()(T x) const { return x > static_cast<T>(0) ? x : static_cast<T>(0); }
};

template<typename T>
struct ReluGradFunctor {
  OF_DEVICE_FUNC explicit ReluGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const { return (y > static_cast<T>(0)) * dy; }
};

template<typename T>
struct SoftShrinkFunctor {
  OF_DEVICE_FUNC explicit SoftShrinkFunctor(double alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x > alpha) return x - alpha;
    if (x < -alpha) return x + alpha;
    return static_cast<T>(0);
  }

  const T alpha;
};

template<typename T>
struct SoftShrinkGradFunctor {
  OF_DEVICE_FUNC explicit SoftShrinkGradFunctor(double alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const {
    return y == static_cast<T>(0) ? static_cast<T>(0) : dy;
  }

  const T alpha;
};

#define REGISTER_SOFTSHRINK_KERNEL(device, dtype)                            \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                       \
      device, "softshrink", SoftShrinkFunctor, dtype, dtype,                 \
      [](user_op::KernelComputeContext* ctx) {                               \
        return SoftShrinkFunctor<dtype>(ctx->Attr<double>("alpha"));         \
      },                                                                     \
      "out", "in");                                                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                      \
      device, "softshrink_grad", SoftShrinkGradFunctor, dtype, dtype, dtype, \
      [](user_op::KernelComputeContext* ctx) {                               \
        return SoftShrinkGradFunctor<dtype>(ctx->Attr<double>("alpha"));     \
      },                                                                     \
      "dx", "y", "dy");

#define REGISTER_ELU_KERNEL(device, dtype)                        \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                            \
      device, "elu", EluFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return EluFunctor<dtype>(ctx->Attr<double>("alpha"));     \
      },                                                          \
      "out", "in");                                               \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                           \
      device, "elu_grad", EluGradFunctor, dtype, dtype, dtype,    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return EluGradFunctor<dtype>(ctx->Attr<double>("alpha")); \
      },                                                          \
      "dx", "x", "dy");

#define REGISTER_LEAKYRELU_KERNEL(device, dtype)                            \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                      \
      device, "leaky_relu", LeakyReluFunctor, dtype, dtype,                 \
      [](user_op::KernelComputeContext* ctx) {                              \
        return LeakyReluFunctor<dtype>(ctx->Attr<float>("alpha"));          \
      },                                                                    \
      "y", "x");                                                            \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                     \
      device, "leaky_relu_grad", LeakyReluGradFunctor, dtype, dtype, dtype, \
      [](user_op::KernelComputeContext* ctx) {                              \
        return LeakyReluGradFunctor<dtype>(ctx->Attr<float>("alpha"));      \
      },                                                                    \
      "dx", "x", "dy");

#define REGISTER_CELU_KERNEL(device, dtype)                        \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                             \
      device, "celu", CeluFunctor, dtype, dtype,                   \
      [](user_op::KernelComputeContext* ctx) {                     \
        return CeluFunctor<dtype>(ctx->Attr<double>("alpha"));     \
      },                                                           \
      "out", "in");                                                \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                            \
      device, "celu_grad", CeluGradFunctor, dtype, dtype, dtype,   \
      [](user_op::KernelComputeContext* ctx) {                     \
        return CeluGradFunctor<dtype>(ctx->Attr<double>("alpha")); \
      },                                                           \
      "dx", "x", "dy");

#define REGISTER_HARDSWISH_KERNEL(device, dtype)                                                   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                             \
      device, "hardswish", HardswishFunctor, dtype, dtype,                                         \
      [](user_op::KernelComputeContext* ctx) { return HardswishFunctor<dtype>(); }, "out", "in");  \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                            \
      device, "hardswish_grad", HardswishGradFunctor, dtype, dtype, dtype,                         \
      [](user_op::KernelComputeContext* ctx) { return HardswishGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_HARDSIGMOID_KERNEL(device, dtype)                                              \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                          \
      device, "hardsigmoid", HardsigmoidFunctor, dtype, dtype,                                  \
      [](user_op::KernelComputeContext* ctx) { return HardsigmoidFunctor<dtype>(); }, "out",    \
      "in");                                                                                    \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                         \
      device, "hardsigmoid_grad", HardsigmoidGradFunctor, dtype, dtype, dtype,                  \
      [](user_op::KernelComputeContext* ctx) { return HardsigmoidGradFunctor<dtype>(); }, "dx", \
      "x", "dy");

#define REGISTER_HARDSHRINK_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("hardshrink")                                                             \
      .SetCreateFn([]() {                                                                        \
        return user_op::NewOpKernel<                                                             \
            UnaryElemwiseXpuKernel<device, HardShrinkFunctor<dtype>, dtype, dtype>>(             \
            [](user_op::KernelComputeContext* ctx) {                                             \
              return HardShrinkFunctor<dtype>(ctx->Attr<double>("lambd"));                       \
            },                                                                                   \
            "out", "in");                                                                        \
      })                                                                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                     \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {  \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                        \
        return Maybe<void>::Ok();                                                                \
      });                                                                                        \
  REGISTER_USER_KERNEL("hardshrink_grad")                                                        \
      .SetCreateFn([]() {                                                                        \
        return user_op::NewOpKernel<                                                             \
            BinaryElemwiseXpuKernel<device, HardShrinkGradFunctor<dtype>, dtype, dtype, dtype>>( \
            [](user_op::KernelComputeContext* ctx) {                                             \
              return HardShrinkGradFunctor<dtype>(ctx->Attr<double>("lambd"));                   \
            },                                                                                   \
            "dx", "y", "dy");                                                                    \
      })                                                                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                     \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {  \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                         \
        return Maybe<void>::Ok();                                                                \
      });

#define REGISTER_HARDTANH_KERNEL(device, dtype)                                                 \
  REGISTER_USER_KERNEL("hardtanh")                                                              \
      .SetCreateFn([]() {                                                                       \
        return user_op::NewOpKernel<                                                            \
            UnaryElemwiseXpuKernel<device, HardtanhFunctor<dtype>, dtype, dtype>>(              \
            [](user_op::KernelComputeContext* ctx) {                                            \
              return HardtanhFunctor<dtype>(ctx->Attr<double>("min_val"),                       \
                                            ctx->Attr<double>("max_val"));                      \
            },                                                                                  \
            "out", "in");                                                                       \
      })                                                                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                     \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });                                                                                       \
  REGISTER_USER_KERNEL("hardtanh_grad")                                                         \
      .SetCreateFn([]() {                                                                       \
        return user_op::NewOpKernel<                                                            \
            BinaryElemwiseXpuKernel<device, HardtanhGradFunctor<dtype>, dtype, dtype, dtype>>(  \
            [](user_op::KernelComputeContext* ctx) {                                            \
              return HardtanhGradFunctor<dtype>(ctx->Attr<double>("min_val"),                   \
                                                ctx->Attr<double>("max_val"));                  \
            },                                                                                  \
            "dx", "y", "dy");                                                                   \
      })                                                                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_MISH_KERNEL(device, dtype)                                                   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                        \
      device, "mish", MishFunctor, dtype, dtype,                                              \
      [](user_op::KernelComputeContext* ctx) { return MishFunctor<dtype>(); }, "out", "in");  \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                       \
      device, "mish_grad", MishGradFunctor, dtype, dtype, dtype,                              \
      [](user_op::KernelComputeContext* ctx) { return MishGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_SILU_KERNEL(device, dtype)                                                   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                        \
      device, "silu", SiluFunctor, dtype, dtype,                                              \
      [](user_op::KernelComputeContext* ctx) { return SiluFunctor<dtype>(); }, "out", "in");  \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                       \
      device, "silu_grad", SiluGradFunctor, dtype, dtype, dtype,                              \
      [](user_op::KernelComputeContext* ctx) { return SiluGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_SELU_KERNEL(device, dtype)                                                   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                        \
      device, "selu", SeluFunctor, dtype, dtype,                                              \
      [](user_op::KernelComputeContext* ctx) { return SeluFunctor<dtype>(); }, "out", "in");  \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                       \
      device, "selu_grad", SeluGradFunctor, dtype, dtype, dtype,                              \
      [](user_op::KernelComputeContext* ctx) { return SeluGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_SOFTSIGN_KERNEL(device, dtype)                                                   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                            \
      device, "softsign", SoftSignFunctor, dtype, dtype,                                          \
      [](user_op::KernelComputeContext* ctx) { return SoftSignFunctor<dtype>(); }, "out", "in");  \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                           \
      device, "softsign_grad", SoftSignGradFunctor, dtype, dtype, dtype,                          \
      [](user_op::KernelComputeContext* ctx) { return SoftSignGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_THRESHOLD_KERNEL(device, dtype)                                \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                          \
      device, "threshold", ThresholdFunctor, dtype, dtype,                      \
      [](user_op::KernelComputeContext* ctx) {                                  \
        return ThresholdFunctor<dtype>(ctx->Attr<double>("threshold_val"),      \
                                       ctx->Attr<double>("value"));             \
      },                                                                        \
      "out", "in");                                                             \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                         \
      device, "threshold_grad", ThresholdGradFunctor, dtype, dtype, dtype,      \
      [](user_op::KernelComputeContext* ctx) {                                  \
        return ThresholdGradFunctor<dtype>(ctx->Attr<double>("threshold_val")); \
      },                                                                        \
      "dx", "x", "dy");

#define REGISTER_SOFTPLUS_KERNEL(device, dtype)                                                   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL(                                                            \
      device, "softplus", SoftplusFunctor, dtype, dtype,                                          \
      [](user_op::KernelComputeContext* ctx) {                                                    \
        return SoftplusFunctor<dtype>(ctx->Attr<double>("beta"), ctx->Attr<double>("threshold")); \
      },                                                                                          \
      "out", "in");                                                                               \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                           \
      device, "softplus_grad", SoftplusGradFunctor, dtype, dtype, dtype,                          \
      [](user_op::KernelComputeContext* ctx) {                                                    \
        return SoftplusGradFunctor<dtype>(ctx->Attr<double>("beta"),                              \
                                          ctx->Attr<double>("threshold"));                        \
      },                                                                                          \
      "dx", "x", "dy");

// For Relu Inplace Proposal Fn.
#define REGISTER_RELU_FORWARD_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("relu")                                                                     \
      .SetCreateFn([]() {                                                                          \
        return user_op::NewOpKernel<                                                               \
            UnaryElemwiseXpuKernel<device, ReluFunctor<dtype>, dtype, dtype>>(                     \
            [](user_op::KernelComputeContext* ctx) { return ReluFunctor<dtype>(); }, "out", "in"); \
      })                                                                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                        \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))           \
      .SetInplaceProposalFn(                                                                       \
          [](const user_op::InferContext&,                                                         \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {               \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                      \
            return Maybe<void>::Ok();                                                              \
          });

#define REGISTER_RELU_BACKWARD_KERNEL(device, dtype)                                           \
  REGISTER_USER_KERNEL("relu_grad")                                                            \
      .SetCreateFn([]() {                                                                      \
        return user_op::NewOpKernel<                                                           \
            BinaryElemwiseXpuKernel<device, ReluGradFunctor<dtype>, dtype, dtype, dtype>>(     \
            [](user_op::KernelComputeContext* ctx) { return ReluGradFunctor<dtype>(); }, "dx", \
            "y", "dy");                                                                        \
      })                                                                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn(                                                                   \
          [](const user_op::InferContext&,                                                     \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {           \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                   \
            return Maybe<void>::Ok();                                                          \
          });

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ACTIVATION_KERNELS_H_
