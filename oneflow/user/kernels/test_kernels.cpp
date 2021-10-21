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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

#ifdef WITH_CUDA

class ReluKernel final : public user_op::OpKernel {
 public:
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    CHECK_NOTNULL(tmp);
    NewKernelUtil<DeviceType::kGPU>::Relu(ctx->device_ctx(), in_blob->shape().elem_cnt(),
                                          in_blob->dptr<float>(), out_blob->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ReluGradKernel final : public user_op::OpKernel {
 public:
  ReluGradKernel() = default;
  ~ReluGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    NewKernelUtil<DeviceType::kGPU>::ReluBackward(
        ctx->device_ctx(), dx_blob->shape().elem_cnt(), y_blob->dptr<float>(),
        y_blob->dptr<float>(), dy_blob->dptr<float>(), dx_blob->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("ccrelu")
    .SetCreateFn<ReluKernel>()
    .SetIsMatchedHob(user_op::HobTrue())
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 10; })
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("ccrelu_grad")
    .SetCreateFn<ReluGradKernel>()
    .SetIsMatchedHob(user_op::HobTrue())
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 10; });

class TestReshapeKernel final : public user_op::OpKernel {
 public:
  TestReshapeKernel() = default;
  ~TestReshapeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                             in_blob->shape().elem_cnt() * sizeof(float));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestReshape")
    .SetCreateFn<TestReshapeKernel>()
    .SetIsMatchedHob(user_op::HobTrue());

class CopyIn2OutKernel final : public user_op::OpKernel {
 public:
  CopyIn2OutKernel() = default;
  ~CopyIn2OutKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                             in_blob->shape().elem_cnt() * sizeof(float));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class TestSourceGpuKernel final : public user_op::OpKernel {
 public:
  TestSourceGpuKernel() = default;
  ~TestSourceGpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {}
};

REGISTER_USER_KERNEL("TestSource")
    .SetCreateFn<TestSourceGpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu")
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 0; });

class TestMultiOutputOrderKernel final : public user_op::OpKernel {
 public:
  TestMultiOutputOrderKernel() = default;
  ~TestMultiOutputOrderKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out1_blob = ctx->Tensor4ArgNameAndIndex("out1", 0);
    user_op::Tensor* out2_blob = ctx->Tensor4ArgNameAndIndex("out2", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out1_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                             in_blob->shape().elem_cnt() * sizeof(float));
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), out2_blob->shape().elem_cnt(), 0.0,
                                          out2_blob->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestMultiOutputOrder")
    .SetCreateFn<TestMultiOutputOrderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kFloat));

class TestMultiInputFwKernel final : public user_op::OpKernel {
 public:
  TestMultiInputFwKernel() = default;
  ~TestMultiInputFwKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1_blob = ctx->Tensor4ArgNameAndIndex("x1", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), y_blob->mut_dptr<char>(), x1_blob->dptr<char>(),
                             x1_blob->shape().elem_cnt() * sizeof(float));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestMultiInput")
    .SetCreateFn<TestMultiInputFwKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("x1", 0) == DataType::kFloat));

class TestMultiInputBwKernel final : public user_op::OpKernel {
 public:
  TestMultiInputBwKernel() = default;
  ~TestMultiInputBwKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x1_diff_blob = ctx->Tensor4ArgNameAndIndex("x1_diff", 0);
    user_op::Tensor* x2_diff_blob = ctx->Tensor4ArgNameAndIndex("x2_diff", 0);
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), x1_diff_blob->shape().elem_cnt(), 1.0,
                                          x1_diff_blob->mut_dptr<float>());
    NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), x2_diff_blob->shape().elem_cnt(), 2.0,
                                          x2_diff_blob->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestMultiInputGrad")
    .SetCreateFn<TestMultiInputBwKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("x1", 0) == DataType::kFloat));

#endif

template<typename T>
class ReluCpuKernel final : public user_op::OpKernel {
 public:
  ReluCpuKernel() = default;
  ~ReluCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    NewKernelUtil<DeviceType::kCPU>::Relu(ctx->device_ctx(), in->shape().elem_cnt(), in->dptr<T>(),
                                          out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("cpu_only_relu_test")
    .SetCreateFn<ReluCpuKernel<float>>()
    .SetIsMatchedHob((user_op::HobDataType("in", 0) == DataType::kFloat)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

class TestSourceKernel final : public user_op::OpKernel {
 public:
  TestSourceKernel() = default;
  ~TestSourceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    for (int i = 0; i < 5; ++i) { *(out_blob->mut_dptr<float>() + i) = static_cast<float>(i); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestSource")
    .SetCreateFn<TestSourceKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat))
    .SetInferTmpSizeFn([](user_op::InferContext*) { return 0; });

class TestSourceMultiGpuFixedOutNumKernel final : public user_op::OpKernel {
 public:
  TestSourceMultiGpuFixedOutNumKernel() = default;
  ~TestSourceMultiGpuFixedOutNumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    for (int i = 0; i < out_blob->shape().elem_cnt(); ++i) {
      *(out_blob->mut_dptr<float>() + i) = static_cast<float>(i);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestSourceMultiGpuFixedOutNum")
    .SetCreateFn<TestSourceMultiGpuFixedOutNumKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

class TestDynamicSourceKernel final : public user_op::OpKernel {
 public:
  TestDynamicSourceKernel() = default;
  ~TestDynamicSourceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    out_blob->mut_shape()->Set(0, 3);
    for (int i = 0; i < 3; ++i) { *(out_blob->mut_dptr<float>() + i) = static_cast<float>(i); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestDynamicSource")
    .SetCreateFn<TestDynamicSourceKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

class TestRandomSourceKernel final : public user_op::OpKernel {
 public:
  TestRandomSourceKernel() = default;
  ~TestRandomSourceKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->Attr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<RandomGenerator<DeviceType::kCPU>>>(
        seed, ctx->device_ctx());
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* random_generator =
        dynamic_cast<OpKernelStateWrapper<RandomGenerator<DeviceType::kCPU>>*>(state);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    random_generator->Mutable()->Uniform<float>(out_blob->shape().elem_cnt(), 0.0, 1.0,
                                                out_blob->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestRandomSource")
    .SetCreateFn<TestRandomSourceKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("out", 0) == DataType::kFloat));

class TestDataTypeAttrKernel final : public user_op::OpKernel {
 public:
  TestDataTypeAttrKernel() = default;
  ~TestDataTypeAttrKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CHECK_EQ(ctx->Attr<DataType>("output_type"),
             ctx->Tensor4ArgNameAndIndex("out", 0)->data_type());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestDataTypeAttr")
    .SetCreateFn<TestDataTypeAttrKernel>()
    .SetIsMatchedHob(user_op::HobTrue());

class TestListDataTypeAndShapeAttrAndStringAttrKernel final : public user_op::OpKernel {
 public:
  TestListDataTypeAndShapeAttrAndStringAttrKernel() = default;
  ~TestListDataTypeAndShapeAttrAndStringAttrKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto& out_shapes = ctx->Attr<std::vector<Shape>>("out_shapes");
    const auto& out_types = ctx->Attr<std::vector<DataType>>("out_types");
    const auto& string_list = ctx->Attr<std::vector<std::string>>("string_list");
    FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
      Shape out_shape_i;
      ctx->Tensor4ArgNameAndIndex("out", i)->shape().ToShape(&out_shape_i);
      CHECK_EQ(out_shapes.at(i), out_shape_i);
      CHECK_EQ(out_types.at(i), ctx->Tensor4ArgNameAndIndex("out", i)->data_type());
    }
    CHECK_GT(string_list.size(), 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("TestListDataTypeAndListShapeAndListStringAttr")
    .SetCreateFn<TestListDataTypeAndShapeAttrAndStringAttrKernel>()
    .SetIsMatchedHob(user_op::HobTrue());

class TestUserOpAttrAutoTypeKernel final : public user_op::OpKernel {
 public:
  TestUserOpAttrAutoTypeKernel() = default;
  ~TestUserOpAttrAutoTypeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { return; }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("test_user_op_attr_auto_type")
    .SetCreateFn<TestUserOpAttrAutoTypeKernel>()
    .SetIsMatchedHob(user_op::HobTrue());

}  // namespace oneflow
