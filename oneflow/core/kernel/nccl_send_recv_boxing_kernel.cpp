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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/core/operator/nccl_send_recv_boxing_op_util.h"

#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700

namespace oneflow {

class NcclSendRecvBoxingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclSendRecvBoxingKernel);
  NcclSendRecvBoxingKernel() = default;
  ~NcclSendRecvBoxingKernel() override = default;

  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec() const {
    return in_tensor_slice_copier_vec_;
  }
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec() const {
    return out_tensor_slice_copier_vec_;
  }
  const std::vector<int64_t>& send_elem_cnts() const { return send_elem_cnts_; }
  const std::vector<int64_t>& recv_elem_cnts() const { return recv_elem_cnts_; }
  const bool has_input() const { return has_input_; }
  const bool has_output() const { return has_output_; }
  ncclComm_t comm() const { return GetOrCreate().comm; }

 private:
  struct Comm {
    Comm(ncclComm_t comm) : comm(comm) {}
    ncclComm_t comm;
  };

  void Init() const {
    ParallelDesc parallel_desc(parallel_conf_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
    ncclComm_t comm = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
    comm_.reset(new Comm(comm));
  }

  const Comm& GetOrCreate() const {
    if (!comm_) { Init(); }
    return *comm_;
  }

  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;

  std::string stream_name_;
  ParallelConf parallel_conf_;
  mutable std::unique_ptr<Comm> comm_;
  bool src_nd_sbp_no_partial_parallel_;
  std::vector<std::shared_ptr<TensorSliceCopier>> in_tensor_slice_copier_vec_;
  std::vector<std::shared_ptr<TensorSliceCopier>> out_tensor_slice_copier_vec_;
  std::vector<int64_t> send_elem_cnts_;
  std::vector<int64_t> recv_elem_cnts_;
  bool has_input_;
  bool has_output_;
};

void NcclSendRecvBoxingKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* buf = ctx->BnInOp2Blob("buf");
  ncclComm_t comm = this->comm();
  cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
  const std::vector<int64_t>& send_elem_cnts = this->send_elem_cnts();
  const std::vector<int64_t>& recv_elem_cnts = this->recv_elem_cnts();
  const int64_t parallel_num = this->kernel_conf().parallel_ctx().parallel_num();
  const DataType data_type = buf->data_type();
  std::vector<void*> send_in_ptr;
  std::vector<void*> recv_out_ptr;
  char* buf_ptr = buf->mut_dptr<char>();
  int64_t offset = 0;
  if (this->has_input()) {
    for (int64_t i = 0; i < parallel_num; ++i) {
      void* send_ptr = reinterpret_cast<void*>(buf_ptr + offset);
      send_in_ptr.push_back(send_ptr);
      offset += send_elem_cnts.at(i) * GetSizeOfDataType(data_type);
    }
  }
  if (this->has_output()) {
    for (int64_t i = 0; i < parallel_num; ++i) {
      void* recv_ptr = reinterpret_cast<void*>(buf_ptr + offset);
      recv_out_ptr.push_back(recv_ptr);
      offset += recv_elem_cnts.at(i) * GetSizeOfDataType(data_type);
    }
  }
  if (this->has_input()) {
    const Blob* in = ctx->BnInOp2Blob("in");
    const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec =
        this->in_tensor_slice_copier_vec();
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (in_tensor_slice_copier_vec.at(i)) {
        in_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), send_in_ptr.at(i), in->dptr());
      }
    }
  }
  OF_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    if (this->has_input() && send_elem_cnts.at(i) != 0) {
      OF_NCCL_CHECK(ncclSend(send_in_ptr.at(i), send_elem_cnts.at(i), GetNcclDataType(data_type), i,
                             comm, cuda_stream));
    }
    if (this->has_output() && recv_elem_cnts.at(i) != 0) {
      OF_NCCL_CHECK(ncclRecv(recv_out_ptr.at(i), recv_elem_cnts.at(i), GetNcclDataType(data_type),
                             i, comm, cuda_stream));
    }
  }
  OF_NCCL_CHECK(ncclGroupEnd());
  if (!this->has_output()) { return; }
  Blob* out = ctx->BnInOp2Blob("out");
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec =
      this->out_tensor_slice_copier_vec();

  if (src_nd_sbp_no_partial_parallel_) {
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i)) {
        out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(), recv_out_ptr.at(i));
      }
    }
  } else {
    std::unique_ptr<ep::primitive::Add> primitive =
        ep::primitive::NewPrimitive<ep::primitive::AddFactory>(ctx->stream()->device_type(),
                                                               out->data_type());
    CHECK(primitive);
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
    CHECK(memset_primitive);
    bool is_first_slice = true;
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i)) {
        if (is_first_slice) {
          is_first_slice = false;
          if (recv_elem_cnts.at(i) != out->shape().elem_cnt()) {
            // if not same shape, memset out
            memset_primitive->Launch(ctx->stream(), out->mut_dptr(), 0,
                                     out->shape().elem_cnt() * GetSizeOfDataType(data_type));
          }
          out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(),
                                                  recv_out_ptr.at(i));
        } else {
          if (recv_elem_cnts.at(i) == out->shape().elem_cnt()) {
            primitive->Launch(ctx->stream(), out->dptr(), recv_out_ptr.at(i), out->mut_dptr(),
                              out->shape().elem_cnt());
          } else {
            void* out_buf = reinterpret_cast<void*>(buf_ptr + offset);
            memset_primitive->Launch(ctx->stream(), out_buf, 0,
                                     out->shape().elem_cnt() * GetSizeOfDataType(data_type));
            out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out_buf, recv_out_ptr.at(i));
            primitive->Launch(ctx->stream(), out->dptr(), out_buf, out->mut_dptr(),
                              out->shape().elem_cnt());
          }
        }
      }
    }
  }
}

void NcclSendRecvBoxingKernel::VirtualKernelInit(KernelContext* ctx) {
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  if (this->op_conf().has_stream_name_hint()) {
    stream_name_ = this->op_conf().stream_name_hint();
  } else {
    stream_name_ = EagerNcclCommMgr::kDefaultStreamName;
  }
  parallel_conf_ = conf.parallel_conf();
  const int64_t parallel_id = this->kernel_conf().parallel_ctx().parallel_id();
  ParallelDesc parallel_desc(parallel_conf_);
  ParallelDesc src_parallel_desc(conf.src_parallel_conf());
  ParallelDesc dst_parallel_desc(conf.dst_parallel_conf());
  const NdSbp& src_nd_sbp = conf.src_nd_sbp();
  const NdSbp& dst_nd_sbp = conf.dst_nd_sbp();
  has_input_ = conf.has_input();
  has_output_ = conf.has_output();
  src_nd_sbp_no_partial_parallel_ = !NdSbpHasPartialParallel(src_nd_sbp);
  const DataType data_type = this->kernel_conf().data_type();
  const DeviceType device_type = parallel_desc.device_type();
  const Shape& logical_shape = Shape(conf.logical_shape());
  const int64_t parallel_num = parallel_desc.parallel_num();

  std::vector<TensorSliceView> src_send_intersections;
  std::vector<TensorSliceView> dst_recv_intersections;
  GetRankSendRecvIntersection(parallel_id, parallel_desc, src_parallel_desc, dst_parallel_desc,
                              src_nd_sbp, dst_nd_sbp, logical_shape, &src_send_intersections,
                              &dst_recv_intersections);
  // if parallel_id exists in src parallel desc, has send
  int64_t src_parallel_id = GetMappedParallelId(parallel_id, parallel_desc, src_parallel_desc);
  if (src_parallel_id != -1) {
    CHECK_EQ(src_send_intersections.size(), parallel_num);
    send_elem_cnts_.resize(parallel_num);
    in_tensor_slice_copier_vec_.resize(parallel_num);
    const TensorSliceView& cur_rank_in_slice = GetTensorSliceView4ParallelId(
        *src_parallel_desc.hierarchy(), src_nd_sbp, logical_shape, src_parallel_id);
    for (int64_t i = 0; i < parallel_num; ++i) {
      const TensorSliceView& intersection = src_send_intersections.at(i);
      if (!intersection.IsEmpty()) {
        send_elem_cnts_.at(i) = intersection.shape().elem_cnt();
        in_tensor_slice_copier_vec_.at(i).reset(
            new TensorSliceCopier(intersection, cur_rank_in_slice, data_type, device_type));
      }
    }
  } else {
    CHECK_EQ(src_send_intersections.size(), 0);
  }

  // if parallel_id exists in src parallel desc, has send
  int64_t dst_parallel_id = GetMappedParallelId(parallel_id, parallel_desc, dst_parallel_desc);
  if (dst_parallel_id != -1) {
    CHECK_EQ(dst_recv_intersections.size(), parallel_num);
    recv_elem_cnts_.resize(parallel_num);
    out_tensor_slice_copier_vec_.resize(parallel_num);
    const TensorSliceView& cur_rank_out_slice = GetTensorSliceView4ParallelId(
        *dst_parallel_desc.hierarchy(), dst_nd_sbp, logical_shape, dst_parallel_id);
    for (int64_t i = 0; i < parallel_num; ++i) {
      const TensorSliceView& intersection = dst_recv_intersections.at(i);
      if (!intersection.IsEmpty()) {
        recv_elem_cnts_.at(i) = intersection.shape().elem_cnt();
        out_tensor_slice_copier_vec_.at(i).reset(
            new TensorSliceCopier(cur_rank_out_slice, intersection, data_type, device_type));
      }
    }
  } else {
    CHECK_EQ(dst_recv_intersections.size(), 0);
  }
}

REGISTER_KERNEL(OperatorConf::kNcclSendRecvBoxingConf, NcclSendRecvBoxingKernel);

REGISTER_SYSTEM_OP_KERNEL_UNIFIED_NCCL_COMM_INIT(OperatorConf::kNcclSendRecvBoxingConf);

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
