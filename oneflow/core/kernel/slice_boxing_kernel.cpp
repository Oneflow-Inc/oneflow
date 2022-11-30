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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {

class SliceBoxingKernel : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingKernel);
  SliceBoxingKernel() = default;
  ~SliceBoxingKernel() override = default;

 protected:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const = 0;

  const std::vector<std::shared_ptr<TensorSliceCopier>>& tensor_slice_copier_vec() const;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;

  std::vector<std::shared_ptr<TensorSliceCopier>> tensor_slice_copier_vec_;
};

class SliceBoxingCopyKernel final : public SliceBoxingKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingCopyKernel);
  SliceBoxingCopyKernel() = default;
  ~SliceBoxingCopyKernel() override = default;

 private:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const override;
  void ForwardDataContent(KernelContext* ctx) const override;
};

class SliceBoxingAddKernel final : public SliceBoxingKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingAddKernel);
  SliceBoxingAddKernel() = default;
  ~SliceBoxingAddKernel() override = default;

 private:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const override;
  void ForwardDataContent(KernelContext* ctx) const override;
};

void SliceBoxingKernel::VirtualKernelInit(KernelContext* ctx) {
  const SliceBoxingConf& conf = GetCustomizedBoxingConf();
  if (/*is_0size_tensor=*/std::any_of(conf.out_shape().dim().begin(), conf.out_shape().dim().end(),
                                      [](int64_t dim) { return dim == 0; })) {
    return;
  }
  const TensorSliceView out_slice(conf.out_slice());
  for (const TensorSliceViewProto& in_slice_proto : conf.in_slice()) {
    const TensorSliceView in_slice(in_slice_proto);
    tensor_slice_copier_vec_.emplace_back(new TensorSliceCopier(
        out_slice, in_slice, this->kernel_conf().data_type(), ctx->stream()->device_type()));
  }
}

const std::vector<std::shared_ptr<TensorSliceCopier>>& SliceBoxingKernel::tensor_slice_copier_vec()
    const {
  return tensor_slice_copier_vec_;
}

const SliceBoxingConf& SliceBoxingCopyKernel::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_copy_conf().slice_boxing_conf();
}

void SliceBoxingCopyKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* out = ctx->BnInOp2Blob("out");
  if (out->shape_view().elem_cnt() == 0) { return; }
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
    this->tensor_slice_copier_vec().at(i)->Copy(ctx->stream(), out, in_i);
  }
}

const SliceBoxingConf& SliceBoxingAddKernel::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_add_conf().slice_boxing_conf();
}

void SliceBoxingAddKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* out = ctx->BnInOp2Blob("out");
  if (out->shape_view().elem_cnt() == 0) { return; }
  std::unique_ptr<ep::primitive::Add> primitive =
      ep::primitive::NewPrimitive<ep::primitive::AddFactory>(ctx->stream()->device_type(),
                                                             out->data_type());
  CHECK(primitive);
  std::unique_ptr<ep::primitive::Memset> memset_primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
  CHECK(memset_primitive);
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
    if (i == 0) {
      if (in_i->shape().NumAxes() == 0 && out->shape().NumAxes() == 0) {
        AutoMemcpy(ctx->stream(), out, in_i);
      } else {
        this->tensor_slice_copier_vec().at(i)->Copy(ctx->stream(), out, in_i);
      }
    } else {
      if (in_i->shape() == out->shape()) {
        primitive->Launch(ctx->stream(), out->dptr(), in_i->dptr(), out->mut_dptr(),
                          out->shape().elem_cnt());
      } else {
        Blob* buf = ctx->BnInOp2Blob("buf");
        memset_primitive->Launch(ctx->stream(), buf->mut_dptr(), 0,
                                 buf->shape().elem_cnt() * GetSizeOfDataType(buf->data_type()));
        this->tensor_slice_copier_vec().at(i)->Copy(ctx->stream(), buf, in_i);
        primitive->Launch(ctx->stream(), out->dptr(), buf->dptr(), out->mut_dptr(),
                          out->shape().elem_cnt());
      }
    }
  }
}

REGISTER_KERNEL(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyKernel);
REGISTER_KERNEL(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddKernel);

}  // namespace oneflow
