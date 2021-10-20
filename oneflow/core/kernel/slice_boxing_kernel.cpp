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
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/primitive/include/add.h"

namespace oneflow {

class SliceBoxingKernel : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingKernel);
  SliceBoxingKernel() = default;
  ~SliceBoxingKernel() override = default;

 protected:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const = 0;
  MemoryCopier* memory_copier() const;
  const std::vector<std::shared_ptr<TensorSliceCopier>>& tensor_slice_copier_vec() const;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;

  std::vector<std::shared_ptr<TensorSliceCopier>> tensor_slice_copier_vec_;
  std::unique_ptr<MemoryCopier> memory_copier_;
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
  memory_copier_.reset(NewDefaultMemoryCopier(ctx->stream_ctx()->device_type()));
  const SliceBoxingConf& conf = GetCustomizedBoxingConf();
  const TensorSliceView out_slice(conf.out_slice());
  for (const TensorSliceViewProto& in_slice_proto : conf.in_slice()) {
    const TensorSliceView in_slice(in_slice_proto);
    tensor_slice_copier_vec_.emplace_back(
        new TensorSliceCopier(out_slice, in_slice, this->kernel_conf().data_type()));
  }
}

MemoryCopier* SliceBoxingKernel::memory_copier() const { return memory_copier_.get(); }

const std::vector<std::shared_ptr<TensorSliceCopier>>& SliceBoxingKernel::tensor_slice_copier_vec()
    const {
  return tensor_slice_copier_vec_;
}

const SliceBoxingConf& SliceBoxingCopyKernel::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_copy_conf().slice_boxing_conf();
}

void SliceBoxingCopyKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* out = ctx->BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
    this->tensor_slice_copier_vec().at(i)->Copy(ctx->device_ctx(), *this->memory_copier(), out,
                                                in_i);
  }
}

const SliceBoxingConf& SliceBoxingAddKernel::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_add_conf().slice_boxing_conf();
}

void SliceBoxingAddKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* out = ctx->BnInOp2Blob("out");
  std::unique_ptr<primitive::Add> primitive = primitive::NewPrimitive<primitive::AddFactory>(
      ctx->stream_ctx()->device_type(), out->data_type());
  CHECK(primitive);
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
    if (i == 0) {
      this->tensor_slice_copier_vec().at(i)->Copy(ctx->device_ctx(), *this->memory_copier(), out,
                                                  in_i);
    } else {
      if (in_i->shape() == out->shape()) {
        primitive->Launch(ctx->stream_ctx(), in_i->dptr(), out->dptr(), out->mut_dptr(),
                          out->shape().elem_cnt());
      } else {
        Blob* buf = ctx->BnInOp2Blob("buf");
        this->tensor_slice_copier_vec().at(i)->Copy(ctx->device_ctx(), *this->memory_copier(), buf,
                                                    in_i);
        primitive->Launch(ctx->stream_ctx(), buf->dptr(), out->dptr(), out->mut_dptr(),
                          out->shape().elem_cnt());
      }
    }
  }
}

REGISTER_KERNEL(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyKernel);
REGISTER_KERNEL(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddKernel);

}  // namespace oneflow
