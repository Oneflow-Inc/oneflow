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
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class DistributeAddKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeAddKernel);
  DistributeAddKernel() = default;
  ~DistributeAddKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
  const Blob* GetInBlob(KernelContext* ctx) const;
};

void DistributeAddKernel::ForwardDataContent(KernelContext* ctx) const {
  AutoMemcpy(ctx->stream(), ctx->BnInOp2Blob("out"), GetInBlob(ctx));
}

const Blob* DistributeAddKernel::GetInBlob(KernelContext* ctx) const {
  const Blob* in_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* cur_blob = ctx->BnInOp2Blob(this->op_attribute().input_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != in_blob) {
      CHECK_ISNULL(in_blob);
      in_blob = cur_blob;
    }
  }
  return in_blob;
}

REGISTER_KERNEL(OperatorConf::kDistributeAddConf, DistributeAddKernel);

class DistributeCloneKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeCloneKernel);
  DistributeCloneKernel() = default;
  ~DistributeCloneKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
  Blob* GetOutBlob(KernelContext* ctx) const;
};

void DistributeCloneKernel::ForwardDataContent(KernelContext* ctx) const {
  AutoMemcpy(ctx->stream(), GetOutBlob(ctx), ctx->BnInOp2Blob("in"));
}

Blob* DistributeCloneKernel::GetOutBlob(KernelContext* ctx) const {
  Blob* out_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().output_bns().size()) {
    Blob* cur_blob = ctx->BnInOp2Blob(this->op_attribute().output_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != out_blob) {
      CHECK_ISNULL(out_blob);
      out_blob = cur_blob;
    }
  }
  return out_blob;
}

REGISTER_KERNEL(OperatorConf::kDistributeCloneConf, DistributeCloneKernel);

class DistributeConcatKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeConcatKernel);
  DistributeConcatKernel() = default;
  ~DistributeConcatKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
  const Blob* GetInBlob(KernelContext* ctx) const;
};

void DistributeConcatKernel::ForwardDataContent(KernelContext* ctx) const {
  AutoMemcpy(ctx->stream(), ctx->BnInOp2Blob("out"), GetInBlob(ctx));
}

const Blob* DistributeConcatKernel::GetInBlob(KernelContext* ctx) const {
  const Blob* in_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* cur_blob = ctx->BnInOp2Blob(this->op_attribute().input_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != in_blob) {
      CHECK_ISNULL(in_blob);
      in_blob = cur_blob;
    }
  }
  return in_blob;
}

REGISTER_KERNEL(OperatorConf::kDistributeConcatConf, DistributeConcatKernel);

class DistributeSplitKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitKernel);
  DistributeSplitKernel() = default;
  ~DistributeSplitKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
  void ForwardShape(KernelContext* ctx) const override;
  Blob* GetOutBlob(KernelContext* ctx) const;
};

void DistributeSplitKernel::ForwardDataContent(KernelContext* ctx) const {
  AutoMemcpy(ctx->stream(), GetOutBlob(ctx), ctx->BnInOp2Blob("in"));
}

void DistributeSplitKernel::ForwardShape(KernelContext* ctx) const {
  Blob* out_blob = GetOutBlob(ctx);
  out_blob->mut_shape_view()->set_shape(ctx->BnInOp2Blob("in")->shape());
}

Blob* DistributeSplitKernel::GetOutBlob(KernelContext* ctx) const {
  Blob* out_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().output_bns().size()) {
    Blob* cur_blob = ctx->BnInOp2Blob(this->op_attribute().output_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != out_blob) {
      CHECK_ISNULL(out_blob);
      out_blob = cur_blob;
    }
  }
  return out_blob;
}

REGISTER_KERNEL(OperatorConf::kDistributeSplitConf, DistributeSplitKernel);

}  // namespace oneflow
