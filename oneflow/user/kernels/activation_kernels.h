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
struct LeakyReluGradFunctor {
  OF_DEVICE_FUNC explicit LeakyReluGradFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const { return (x > 0) ? dy : dy * alpha; }
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
struct CeluGradFunctor {
  OF_DEVICE_FUNC explicit CeluGradFunctor(float alpha) : inv_alpha(1.0f / alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? dy : dy * static_cast<T>(exp(x * inv_alpha));
  }
  const T inv_alpha;
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
struct HardsigmoidGradFunctor {
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(-3) && x < static_cast<T>(3)) ? dy / static_cast<T>(6)
                                                             : static_cast<T>(0);
  }
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
struct SiluGradFunctor {
  OF_DEVICE_FUNC explicit SiluGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    T sig = static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
    return dy * (sig * (static_cast<T>(1) + x * (static_cast<T>(1) - sig)));
  }
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
struct SoftSignGradFunctor {
  OF_DEVICE_FUNC explicit SoftSignGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    T val = (static_cast<T>(1) + abs(x));
    return dy / (val * val);
  }
};

template<typename T>
struct ThresholdGradFunctor {
  OF_DEVICE_FUNC explicit ThresholdGradFunctor(double threshold) : threshold(threshold) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const { return (x > threshold) ? dy : static_cast<T>(0); }
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
struct ReluGradFunctor {
  OF_DEVICE_FUNC explicit ReluGradFunctor() {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const { return (y > static_cast<T>(0)) * dy; }
};

template<typename T>
struct SoftShrinkGradFunctor {
  OF_DEVICE_FUNC explicit SoftShrinkGradFunctor(double alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const {
    return y == static_cast<T>(0) ? static_cast<T>(0) : dy;
  }

  const T alpha;
};

namespace {
auto UnaryPrimitiveExists(ep::primitive::UnaryOp op, const std::string& output_name,
                          const std::string& input_name) {
  return hob::make_custom("PrimitiveExists", [=](const user_op::KernelRegContext& ctx) {
    const user_op::TensorDesc* src = ctx.TensorDesc4ArgNameAndIndex(input_name, 0);
    const user_op::TensorDesc* dst = ctx.TensorDesc4ArgNameAndIndex(output_name, 0);
    auto primitive = ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
        ctx.device_type(), op, src->data_type(), dst->data_type());
    return primitive.operator bool();
  });
}
}  // namespace

#define REGISTER_SOFTSHRINK_FORWARD_KERNEL()                                                 \
  REGISTER_USER_KERNEL("softshrink")                                                         \
      .SetCreateFn([]() {                                                                    \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                   \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                            \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);     \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);    \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(    \
                  ctx->device_type(), ep::primitive::UnaryOp::kSoftShrink, src->data_type(), \
                  dst->data_type(), ctx->Attr<double>("alpha"));                             \
            });                                                                              \
      })                                                                                     \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSoftShrink, "out", "in"));

#define REGISTER_SOFTSHRINK_BACKWARD_KERNEL(device, dtype)                   \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                      \
      device, "softshrink_grad", SoftShrinkGradFunctor, dtype, dtype, dtype, \
      [](user_op::KernelComputeContext* ctx) {                               \
        return SoftShrinkGradFunctor<dtype>(ctx->Attr<double>("alpha"));     \
      },                                                                     \
      "dx", "y", "dy");

#define REGISTER_ELU_FORWARD_KERNEL()                                                     \
  REGISTER_USER_KERNEL("elu")                                                             \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kElu, src->data_type(),     \
                  dst->data_type(), ctx->Attr<double>("alpha"));                          \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kElu, "out", "in"));

#define REGISTER_ELU_BACKWARD_KERNEL(device, dtype)               \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                           \
      device, "elu_grad", EluGradFunctor, dtype, dtype, dtype,    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return EluGradFunctor<dtype>(ctx->Attr<double>("alpha")); \
      },                                                          \
      "dx", "x", "dy");

#define REGISTER_GELU_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("gelu")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kGelu, src->data_type(),    \
                  dst->data_type());                                                      \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kGelu, "out", "in"));

#define REGISTER_LEAKYRELU_FORWARD_KERNEL()                                                 \
  REGISTER_USER_KERNEL("leaky_relu")                                                        \
      .SetCreateFn([]() {                                                                   \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                  \
            "y", "x", [](user_op::KernelComputeContext* ctx) {                              \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);     \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);     \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(   \
                  ctx->device_type(), ep::primitive::UnaryOp::kLeakyRelu, src->data_type(), \
                  dst->data_type(), ctx->Attr<float>("alpha"));                             \
            });                                                                             \
      })                                                                                    \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kLeakyRelu, "y", "x"));

#define REGISTER_LEAKYRELU_BACKWARD_KERNEL(device, dtype)                   \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                     \
      device, "leaky_relu_grad", LeakyReluGradFunctor, dtype, dtype, dtype, \
      [](user_op::KernelComputeContext* ctx) {                              \
        return LeakyReluGradFunctor<dtype>(ctx->Attr<float>("alpha"));      \
      },                                                                    \
      "dx", "x", "dy");

#define REGISTER_CELU_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("celu")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kCelu, src->data_type(),    \
                  dst->data_type(), ctx->Attr<double>("alpha"));                          \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kCelu, "out", "in"));

#define REGISTER_CELU_BACKWARD_KERNEL(device, dtype)               \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                            \
      device, "celu_grad", CeluGradFunctor, dtype, dtype, dtype,   \
      [](user_op::KernelComputeContext* ctx) {                     \
        return CeluGradFunctor<dtype>(ctx->Attr<double>("alpha")); \
      },                                                           \
      "dx", "x", "dy");

#define REGISTER_HARDSWISH_FORWARD_KERNEL()                                                 \
  REGISTER_USER_KERNEL("hardswish")                                                         \
      .SetCreateFn([]() {                                                                   \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                  \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                           \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);    \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);   \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(   \
                  ctx->device_type(), ep::primitive::UnaryOp::kHardSwish, src->data_type(), \
                  dst->data_type());                                                        \
            });                                                                             \
      })                                                                                    \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardSwish, "out", "in"));

#define REGISTER_HARDSWISH_BACKWARD_KERNEL(device, dtype)                                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                            \
      device, "hardswish_grad", HardswishGradFunctor, dtype, dtype, dtype,                         \
      [](user_op::KernelComputeContext* ctx) { return HardswishGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_HARDSIGMOID_FORWARD_KERNEL()                                                 \
  REGISTER_USER_KERNEL("hardsigmoid")                                                         \
      .SetCreateFn([]() {                                                                     \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                    \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                             \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);      \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);     \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(     \
                  ctx->device_type(), ep::primitive::UnaryOp::kHardSigmoid, src->data_type(), \
                  dst->data_type());                                                          \
            });                                                                               \
      })                                                                                      \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardSigmoid, "out", "in"));

#define REGISTER_HARDSIGMOID_BACKWARD_KERNEL(device, dtype)                                     \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                         \
      device, "hardsigmoid_grad", HardsigmoidGradFunctor, dtype, dtype, dtype,                  \
      [](user_op::KernelComputeContext* ctx) { return HardsigmoidGradFunctor<dtype>(); }, "dx", \
      "x", "dy");

#define REGISTER_HARDSHRINK_FORWARD_KERNEL()                                                   \
  REGISTER_USER_KERNEL("hardshrink")                                                           \
      .SetCreateFn([]() {                                                                      \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                     \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                              \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);       \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);      \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(      \
                  ctx->device_type(), ep::primitive::UnaryOp::kHardShrink, src->data_type(),   \
                  dst->data_type(), ctx->Attr<double>("lambd"));                               \
            });                                                                                \
      })                                                                                       \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardShrink, "out", "in")) \
      .SetInplaceProposalFn(                                                                   \
          [](const user_op::InferContext&,                                                     \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {           \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                  \
            return Maybe<void>::Ok();                                                          \
          });

#define REGISTER_HARDSHRINK_BACKWARD_KERNEL(device, dtype)                                       \
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
      .SetInplaceProposalFn(                                                                     \
          [](const user_op::InferContext&,                                                       \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {             \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                     \
            return Maybe<void>::Ok();                                                            \
          });

#define REGISTER_HARDTANH_FORWARD_KERNEL()                                                       \
  REGISTER_USER_KERNEL("hardtanh")                                                               \
      .SetCreateFn([]() {                                                                        \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                       \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                                \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);         \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);        \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(        \
                  ctx->device_type(), ep::primitive::UnaryOp::kHardTanh, src->data_type(),       \
                  dst->data_type(), ctx->Attr<double>("min_val"), ctx->Attr<double>("max_val")); \
            });                                                                                  \
      })                                                                                         \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardTanh, "out", "in"))     \
      .SetInplaceProposalFn(                                                                     \
          [](const user_op::InferContext&,                                                       \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {             \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                    \
            return Maybe<void>::Ok();                                                            \
          });

#define REGISTER_HARDTANH_BACKWARD_KERNEL(device, dtype)                                       \
  REGISTER_USER_KERNEL("hardtanh_grad")                                                        \
      .SetCreateFn([]() {                                                                      \
        return user_op::NewOpKernel<                                                           \
            BinaryElemwiseXpuKernel<device, HardtanhGradFunctor<dtype>, dtype, dtype, dtype>>( \
            [](user_op::KernelComputeContext* ctx) {                                           \
              return HardtanhGradFunctor<dtype>(ctx->Attr<double>("min_val"),                  \
                                                ctx->Attr<double>("max_val"));                 \
            },                                                                                 \
            "dx", "y", "dy");                                                                  \
      })                                                                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn(                                                                   \
          [](const user_op::InferContext&,                                                     \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {           \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                   \
            return Maybe<void>::Ok();                                                          \
          });

#define REGISTER_TANH_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("tanh")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "y", "x", [](user_op::KernelComputeContext* ctx) {                            \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);   \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);   \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kTanh, src->data_type(),    \
                  dst->data_type());                                                      \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kTanh, "y", "x"));

#define REGISTER_MISH_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("mish")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kMish, src->data_type(),    \
                  dst->data_type());                                                      \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kMish, "out", "in"));

#define REGISTER_MISH_BACKWARD_KERNEL(device, dtype)                                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                       \
      device, "mish_grad", MishGradFunctor, dtype, dtype, dtype,                              \
      [](user_op::KernelComputeContext* ctx) { return MishGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_SILU_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("silu")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kSilu, src->data_type(),    \
                  dst->data_type());                                                      \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSilu, "out", "in"));

#define REGISTER_SILU_BACKWARD_KERNEL(device, dtype)                                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                       \
      device, "silu_grad", SiluGradFunctor, dtype, dtype, dtype,                              \
      [](user_op::KernelComputeContext* ctx) { return SiluGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_SELU_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("selu")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                         \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);  \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0); \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kSelu, src->data_type(),    \
                  dst->data_type());                                                      \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSelu, "out", "in"));

#define REGISTER_SELU_BACKWARD_KERNEL(device, dtype)                                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                       \
      device, "selu_grad", SeluGradFunctor, dtype, dtype, dtype,                              \
      [](user_op::KernelComputeContext* ctx) { return SeluGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_SOFTSIGN_FORWARD_KERNEL()                                                 \
  REGISTER_USER_KERNEL("softsign")                                                         \
      .SetCreateFn([]() {                                                                  \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                 \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                          \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);   \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);  \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(  \
                  ctx->device_type(), ep::primitive::UnaryOp::kSoftSign, src->data_type(), \
                  dst->data_type());                                                       \
            });                                                                            \
      })                                                                                   \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSoftSign, "out", "in"));

#define REGISTER_SOFTSIGN_BACKWARD_KERNEL(device, dtype)                                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                           \
      device, "softsign_grad", SoftSignGradFunctor, dtype, dtype, dtype,                          \
      [](user_op::KernelComputeContext* ctx) { return SoftSignGradFunctor<dtype>(); }, "dx", "x", \
      "dy");

#define REGISTER_THRESHOLD_FORWARD_KERNEL()                                                 \
  REGISTER_USER_KERNEL("threshold")                                                         \
      .SetCreateFn([]() {                                                                   \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                  \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                           \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);    \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);   \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(   \
                  ctx->device_type(), ep::primitive::UnaryOp::kThreshold, src->data_type(), \
                  dst->data_type(), ctx->Attr<double>("threshold_val"),                     \
                  ctx->Attr<double>("value"));                                              \
            });                                                                             \
      })                                                                                    \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kThreshold, "out", "in"));

#define REGISTER_THRESHOLD_BACKWARD_KERNEL(device, dtype)                       \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                         \
      device, "threshold_grad", ThresholdGradFunctor, dtype, dtype, dtype,      \
      [](user_op::KernelComputeContext* ctx) {                                  \
        return ThresholdGradFunctor<dtype>(ctx->Attr<double>("threshold_val")); \
      },                                                                        \
      "dx", "x", "dy");

#define REGISTER_SOFTPLUS_FORWARD_KERNEL()                                                      \
  REGISTER_USER_KERNEL("softplus")                                                              \
      .SetCreateFn([]() {                                                                       \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                      \
            "out", "in", [](user_op::KernelComputeContext* ctx) {                               \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);        \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);       \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(       \
                  ctx->device_type(), ep::primitive::UnaryOp::kSoftPlus, src->data_type(),      \
                  dst->data_type(), ctx->Attr<double>("beta"), ctx->Attr<double>("threshold")); \
            });                                                                                 \
      })                                                                                        \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSoftPlus, "out", "in"));

#define REGISTER_SOFTPLUS_BACKWARD_KERNEL(device, dtype)                   \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                    \
      device, "softplus_grad", SoftplusGradFunctor, dtype, dtype, dtype,   \
      [](user_op::KernelComputeContext* ctx) {                             \
        return SoftplusGradFunctor<dtype>(ctx->Attr<double>("beta"),       \
                                          ctx->Attr<double>("threshold")); \
      },                                                                   \
      "dx", "x", "dy");

#define REGISTER_RELU_FORWARD_KERNEL()                                                    \
  REGISTER_USER_KERNEL("relu")                                                            \
      .SetCreateFn([]() {                                                                 \
        return user_op::NewOpKernel<UnaryPrimitiveKernel>(                                \
            "y", "x", [](user_op::KernelComputeContext* ctx) {                            \
              const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);   \
              const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);   \
              return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>( \
                  ctx->device_type(), ep::primitive::UnaryOp::kRelu, src->data_type(),    \
                  dst->data_type());                                                      \
            });                                                                           \
      })                                                                                  \
      .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kRelu, "y", "x"))     \
      .SetInplaceProposalFn(                                                              \
          [](const user_op::InferContext&,                                                \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {      \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                \
            return Maybe<void>::Ok();                                                     \
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
