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
#include "oneflow/core/ep/include/primitive/memcpy.h"

namespace oneflow {

class CopyHdKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;
  void ForwardHeader(KernelContext* ctx) const override;

  std::unique_ptr<ep::primitive::Memcpy> primitive_;
};

void CopyHdKernel::VirtualKernelInit(KernelContext* ctx) {
  CHECK(this->op_conf().has_copy_hd_conf());
  const CopyHdOpConf& copy_hd_conf = this->op_conf().copy_hd_conf();
  ep::primitive::MemcpyKind kind{};
  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
    kind = ep::primitive::MemcpyKind::kHtoD;
  } else if (copy_hd_conf.type() == CopyHdOpConf::D2H) {
    kind = ep::primitive::MemcpyKind::kDtoH;
  } else {
    UNIMPLEMENTED();
  }
  primitive_ =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->stream()->device_type(), kind);
  CHECK(primitive_);
}

void CopyHdKernel::ForwardDataContent(KernelContext* ctx) const {
  const Blob* in_blob = ctx->BnInOp2Blob(op_attribute().input_bns(0));
  Blob* out_blob = ctx->BnInOp2Blob(op_attribute().output_bns(0));
  const size_t body_byte_size = in_blob->ByteSizeOfBlobBody();
  CHECK_EQ(out_blob->ByteSizeOfBlobBody(), body_byte_size);
  primitive_->Launch(ctx->stream(), out_blob->mut_dptr(), in_blob->dptr(), body_byte_size);
}

void CopyHdKernel::ForwardHeader(KernelContext* ctx) const {
  ctx->BnInOp2Blob("out")->CopyHeaderFrom(ctx->BnInOp2Blob("in"));
}

REGISTER_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);

}  // namespace oneflow
