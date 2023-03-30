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
#include "oneflow/user/kernels/collective_communication/include/all_gather.h"
#include "oneflow/cambricon/collective_communication/mlu_communication_context.h"
#include "oneflow/cambricon/collective_communication/cncl_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {

namespace ccl {

class MluAllGather final : public AllGather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluAllGather);
  MluAllGather() : cncl_datatype_(), size_of_element_(0) {}
  ~MluAllGather() = default;

  void Init(DataType datatype) override {
    this->cncl_datatype_ = cnclChar;
    this->size_of_element_ = GetSizeOfDataType(datatype);
  }

  void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
              const std::shared_ptr<CommunicationContext>& communication_ctx) const override {
    const auto& mlu_communication_ctx =
        std::dynamic_pointer_cast<MluCommunicationContext>(communication_ctx);
    CHECK(mlu_communication_ctx) << kOfBugIssueUploadPrompt;
    OF_CNCL_CHECK(cnclAllGather(in, out, elem_cnt * size_of_element_, cncl_datatype_,
                                mlu_communication_ctx->cncl_comm(),
                                stream->As<ep::MluStream>()->mlu_stream()));
  }

 private:
  cnclDataType_t cncl_datatype_;
  size_t size_of_element_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kMLU, AllGather, MluAllGather);

}  // namespace ccl

}  // namespace oneflow
