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
#include "oneflow/user/kernels/collective_communication/include/reduce_scatter.h"
#include "oneflow/cambricon/collective_communication/mlu_communication_context.h"
#include "oneflow/cambricon/collective_communication/cncl_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {

namespace ccl {

namespace {

inline cnclReduceOp_t GetCnclReduceType(ReduceType reduce_type) {
  switch (reduce_type) {
#define CNCL_REDUCE_TYPE_CASE(dtype) \
  case ReduceType::k##dtype: return cnclReduceOp_t::cncl##dtype
    CNCL_REDUCE_TYPE_CASE(Sum);
    CNCL_REDUCE_TYPE_CASE(Max);
    default: PRINT_BUG_PROMPT_AND_ABORT();
  }
}

}  // namespace

class MluReduceScatter final : public ReduceScatter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluReduceScatter);
  MluReduceScatter() : cncl_datatype_(), cncl_reduce_op_() {}
  ~MluReduceScatter() = default;

  void Init(DataType datatype, ReduceType reduce_type) override {
    this->cncl_datatype_ = GetCnclDataType(datatype);
    this->cncl_reduce_op_ = GetCnclReduceType(reduce_type);
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& mlu_communication_ctx =
        std::dynamic_pointer_cast<MluCommunicationContext>(communication_ctx);
    CHECK(mlu_communication_ctx) << kOfBugIssueUploadPrompt;
    OF_CNCL_CHECK(cnclReduceScatter(in, out, elem_cnt, cncl_datatype_, cncl_reduce_op_,
                                    mlu_communication_ctx->cncl_comm(),
                                    stream->As<ep::MluStream>()->mlu_stream()));
  }

 private:
  cnclDataType_t cncl_datatype_;
  cnclReduceOp_t cncl_reduce_op_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kMLU, ReduceScatter, MluReduceScatter);

}  // namespace ccl

}  // namespace oneflow
