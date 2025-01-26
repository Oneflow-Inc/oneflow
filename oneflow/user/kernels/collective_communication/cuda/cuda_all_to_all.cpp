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
#include "oneflow/user/kernels/collective_communication/include/all_to_all.h"
#include "oneflow/user/kernels/collective_communication/cuda/cuda_communication_context.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

namespace ccl {

class CudaAllToAll final : public AllToAll {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaAllToAll);
  CudaAllToAll()
      : send_dtype_(), recv_dtype_(), nccl_send_dtype_(), nccl_recv_dtype_(), rank_count_(0) {}
  ~CudaAllToAll() = default;

  void Init(DataType send_dtype, DataType recv_dtype, size_t parallel_num) override {
    this->send_dtype_ = send_dtype;
    this->recv_dtype_ = recv_dtype;
    this->nccl_send_dtype_ = GetNcclDataType(send_dtype);
    this->nccl_recv_dtype_ = GetNcclDataType(recv_dtype);
    this->rank_count_ = parallel_num;
  }

  void Launch(ep::Stream* stream, void* send, int64_t send_count, void* recv, int64_t recv_count,
              const ccl::CclComm& ccl_comm) const override {
    ncclComm_t* nccl_comm = reinterpret_cast<ncclComm_t*>(ccl_comm.getComm());
    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t i = 0; i < this->rank_count_; ++i) {
      if (send_count > 0) {
        char* send_ptr = static_cast<char*>(send) + send_offset;
        OF_NCCL_CHECK(ncclSend(send_ptr, send_count, this->nccl_send_dtype_, i, *nccl_comm,
                               stream->As<ep::CudaStream>()->cuda_stream()));
      }
      send_offset += send_count * GetSizeOfDataType(this->send_dtype_);
      if (recv_count) {
        char* recv_ptr = static_cast<char*>(recv) + recv_offset;
        OF_NCCL_CHECK(ncclRecv(recv_ptr, recv_count, this->nccl_recv_dtype_, i, *nccl_comm,
                               stream->As<ep::CudaStream>()->cuda_stream()));
      }
      recv_offset += recv_count * GetSizeOfDataType(this->recv_dtype_);
    }
    OF_NCCL_CHECK(ncclGroupEnd());
  }

  void Launch(ep::Stream* stream, void* send, const void* send_counts, const void* send_offsets,
              void* recv, const void* recv_counts, const void* recv_offsets,
              const ccl::CclComm& ccl_comm) const override {
    ncclComm_t* nccl_comm = reinterpret_cast<ncclComm_t*>(ccl_comm.getComm());
    int64_t* send_counts_ptr = static_cast<int64_t*>(const_cast<void*>(send_counts));
    int64_t* recv_counts_ptr = static_cast<int64_t*>(const_cast<void*>(recv_counts));
    int64_t* send_offsets_ptr = static_cast<int64_t*>(const_cast<void*>(send_offsets));
    int64_t* recv_offsets_ptr = static_cast<int64_t*>(const_cast<void*>(recv_offsets));
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t i = 0; i < this->rank_count_; ++i) {
      uint64_t send_offset = static_cast<uint64_t>(send_offsets_ptr[i]);
      uint64_t send_count = static_cast<uint64_t>(send_counts_ptr[i]);
      char* send_ptr = static_cast<char*>(send) + send_offset;
      if (send_count > 0) {
        OF_NCCL_CHECK(ncclSend(send_ptr, send_count, this->nccl_send_dtype_, i, *nccl_comm,
                               stream->As<ep::CudaStream>()->cuda_stream()));
      }

      uint64_t recv_offset = static_cast<uint64_t>(recv_offsets_ptr[i]);
      uint64_t recv_count = static_cast<uint64_t>(recv_counts_ptr[i]);
      char* recv_ptr = static_cast<char*>(recv) + recv_offset;
      if (recv_count > 0) {
        OF_NCCL_CHECK(ncclRecv(recv_ptr, recv_count, this->nccl_recv_dtype_, i, *nccl_comm,
                               stream->As<ep::CudaStream>()->cuda_stream()));
      }
    }
    OF_NCCL_CHECK(ncclGroupEnd());
  }

 private:
  DataType send_dtype_;
  DataType recv_dtype_;
  ncclDataType_t nccl_send_dtype_;
  ncclDataType_t nccl_recv_dtype_;
  size_t rank_count_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCUDA, AllToAll, CudaAllToAll);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA
