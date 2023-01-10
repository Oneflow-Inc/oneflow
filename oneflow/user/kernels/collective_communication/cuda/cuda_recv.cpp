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
#include "oneflow/user/kernels/collective_communication/include/recv.h"
#include "oneflow/user/kernels/collective_communication/cuda/cuda_send_recv_util.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace ccl {

class CudaRecv final : public Recv {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRecv);
  CudaRecv() : nccl_datatype_() {}
  ~CudaRecv() = default;

  void Init(DataType datatype) override { this->nccl_datatype_ = GetNcclDataType(datatype); }

  void Launch(ep::Stream* stream, void* out, size_t elem_cnt, int64_t src) const override {
#if HAS_NCCL_SEND_RECV
    const auto& comm_and_peer_rank = GetNcclCommAndPeerNcclRank(src);
    OF_NCCL_CHECK(ncclRecv(out, elem_cnt, nccl_datatype_, comm_and_peer_rank.second,
                           comm_and_peer_rank.first, stream->As<ep::CudaStream>()->cuda_stream()));
#else
    UNIMPLEMENTED() << "GPU recv is only supported when nccl version >= 2.7"
#endif  // HAS_NCCL_SEND_RECV
  }

 private:
  ncclDataType_t nccl_datatype_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCUDA, Recv, CudaRecv);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
