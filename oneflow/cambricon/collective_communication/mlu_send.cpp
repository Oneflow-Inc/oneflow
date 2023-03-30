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
#include "oneflow/user/kernels/collective_communication/include/send.h"
#include "oneflow/cambricon/collective_communication/mlu_send_recv_util.h"
#include "oneflow/cambricon/collective_communication/cncl_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {

namespace ccl {

class MluSend final : public Send {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluSend);
  MluSend() : cncl_datatype_(), size_of_element_(0) {}
  ~MluSend() = default;

  void Init(DataType datatype) override {
    this->cncl_datatype_ = cnclChar;
    this->size_of_element_ = GetSizeOfDataType(datatype);
  }

  void Launch(ep::Stream* stream, const void* in, size_t elem_cnt, int64_t dst) const override {
    const auto& comm_and_peer_rank = GetCnclCommAndPeerCnclRank(dst);
    OF_CNCL_CHECK(cnclSend(const_cast<void*>(in), elem_cnt * size_of_element_, cncl_datatype_,
                           comm_and_peer_rank.second, comm_and_peer_rank.first,
                           stream->As<ep::MluStream>()->mlu_stream()));
  }

 private:
  cnclDataType_t cncl_datatype_;
  size_t size_of_element_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kMLU, Send, MluSend);

}  // namespace ccl

}  // namespace oneflow
