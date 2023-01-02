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
#ifdef WITH_CUDA
#include "oneflow/user/kernels/collective_communication/include/send.h"
#include "oneflow/user/kernels/collective_communication/cuda/cuda_send_recv_util.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace ccl {

class CudaSend final : public Send {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaSend);
  CudaSend() : nccl_datatype_() {}
  ~CudaSend() = default;

  void Init(DataType datatype) override { this->nccl_datatype_ = GetNcclDataType(datatype); }

  void Launch(ep::Stream* stream, const void* in, size_t elem_cnt, int64_t dst) const override {
#if HAS_NCCL_SEND_RECV
    const auto& comm_and_peer_rank = GetNcclCommAndPeerNcclRank(dst);
    OF_NCCL_CHECK(ncclSend(in, elem_cnt, nccl_datatype_, comm_and_peer_rank.second,
                           comm_and_peer_rank.first, stream->As<ep::CudaStream>()->cuda_stream()));
#else
    UNIMPLEMENTED() << "GPU send is only supported when nccl version >= 2.7"
#endif  // HAS_NCCL_SEND_RECV
  }

 private:
  ncclDataType_t nccl_datatype_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCUDA, Send, CudaSend);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
