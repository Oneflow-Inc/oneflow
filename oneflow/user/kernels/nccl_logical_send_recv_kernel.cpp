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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/user/ops/nccl_logical_util.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/core/operator/nccl_send_recv_boxing_op_util.h"

#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700

namespace oneflow {

class NcclLogicalSendRecvState final : public user_op::OpKernelState {
 public:
  explicit NcclLogicalSendRecvState(user_op::KernelInitContext* ctx);
  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec() const {
    return in_tensor_slice_copier_vec_;
  }
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec() const {
    return out_tensor_slice_copier_vec_;
  }
  bool src_nd_sbp_has_no_partial_parallel() const { return src_nd_sbp_no_partial_parallel_; }
  const std::vector<int64_t>& send_elem_cnts() const { return send_elem_cnts_; }
  const std::vector<int64_t>& recv_elem_cnts() const { return recv_elem_cnts_; }
  ncclComm_t comm() const { return GetOrCreateComm().comm; }

 private:
  struct Comm {
    explicit Comm(ncclComm_t comm) : comm(comm) {}
    ncclComm_t comm;
  };
  void InitComm() const;
  const Comm& GetOrCreateComm() const {
    if (!comm_) { InitComm(); }
    return *comm_;
  }

  std::string stream_name_;
  std::unique_ptr<ParallelDesc> parallel_desc_;
  mutable std::unique_ptr<Comm> comm_;
  bool src_nd_sbp_no_partial_parallel_;
  std::vector<std::shared_ptr<TensorSliceCopier>> in_tensor_slice_copier_vec_;
  std::vector<std::shared_ptr<TensorSliceCopier>> out_tensor_slice_copier_vec_;
  std::vector<int64_t> send_elem_cnts_;
  std::vector<int64_t> recv_elem_cnts_;
};

NcclLogicalSendRecvState::NcclLogicalSendRecvState(user_op::KernelInitContext* ctx)
    : stream_name_(EagerNcclCommMgr::kDefaultStreamName) {
  if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  parallel_desc_ = std::make_unique<ParallelDesc>(ctx->parallel_desc());
  NdSbp src_nd_sbp;
  CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_nd_sbp", &src_nd_sbp));
  NdSbp dst_nd_sbp;
  CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_nd_sbp", &dst_nd_sbp));
  const auto& parallel_hierarchy = parallel_desc_->hierarchy();
  src_nd_sbp_no_partial_parallel_ = !NdSbpHasPartialParallel(src_nd_sbp);
  CHECK_EQ(src_nd_sbp.sbp_parallel_size(), parallel_hierarchy->NumAxes());
  CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), parallel_hierarchy->NumAxes());
  const user_op::TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
  const DataType data_type = in_logical_desc->data_type();
  const Shape& logical_shape = Shape(in_logical_desc->shape());
  const DeviceType device_type = parallel_desc_->device_type();
  const int64_t parallel_num = parallel_desc_->parallel_num();

  std::vector<TensorSliceView> src_send_intersections;
  std::vector<TensorSliceView> dst_recv_intersections;
  GetRankSendRecvIntersection(parallel_id, /*merge_parallel_desc=*/*parallel_desc_,
                              /*in_parallel_desc=*/*parallel_desc_,
                              /*out_parallel_desc=*/*parallel_desc_, src_nd_sbp, dst_nd_sbp,
                              logical_shape, &src_send_intersections, &dst_recv_intersections);

  CHECK_EQ(src_send_intersections.size(), parallel_num);
  send_elem_cnts_.resize(parallel_num);
  in_tensor_slice_copier_vec_.resize(parallel_num);
  const TensorSliceView& cur_rank_in_slice =
      GetTensorSliceView4ParallelId(*parallel_hierarchy, src_nd_sbp, logical_shape, parallel_id);
  for (int64_t i = 0; i < parallel_num; ++i) {
    const TensorSliceView& intersection = src_send_intersections.at(i);
    if (!intersection.IsEmpty()) {
      send_elem_cnts_.at(i) = intersection.shape().elem_cnt();
      in_tensor_slice_copier_vec_.at(i).reset(
          new TensorSliceCopier(intersection, cur_rank_in_slice, data_type, device_type));
    }
  }

  CHECK_EQ(dst_recv_intersections.size(), parallel_num);
  recv_elem_cnts_.resize(parallel_num);
  out_tensor_slice_copier_vec_.resize(parallel_num);
  const TensorSliceView& cur_rank_out_slice =
      GetTensorSliceView4ParallelId(*parallel_hierarchy, dst_nd_sbp, logical_shape, parallel_id);
  for (int64_t i = 0; i < parallel_num; ++i) {
    const TensorSliceView& intersection = dst_recv_intersections.at(i);
    if (!intersection.IsEmpty()) {
      recv_elem_cnts_.at(i) = intersection.shape().elem_cnt();
      out_tensor_slice_copier_vec_.at(i).reset(
          new TensorSliceCopier(cur_rank_out_slice, intersection, data_type, device_type));
    }
  }
}

void NcclLogicalSendRecvState::InitComm() const {
  std::set<std::pair<int64_t, int64_t>> device_set;
  for (int64_t parallel_id = 0; parallel_id < parallel_desc_->parallel_num(); ++parallel_id) {
    int64_t machine_id = CHECK_JUST(parallel_desc_->MachineId4ParallelId(parallel_id));
    int64_t device_id = CHECK_JUST(parallel_desc_->DeviceId4ParallelId(parallel_id));
    device_set.emplace(std::make_pair(machine_id, device_id));
  }
  EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
  ncclComm_t comm = nullptr;
  comm = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
  comm_.reset(new Comm(comm));
}

class NcclLogicalSendRecv final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalSendRecv);
  NcclLogicalSendRecv() = default;
  ~NcclLogicalSendRecv() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogicalSendRecvState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchNcclLogicalKernel();
  }
};

void NcclLogicalSendRecv::Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                                  const user_op::OpKernelCache*) const {
  auto* kernel_state = dynamic_cast<NcclLogicalSendRecvState*>(state);
  CHECK_NOTNULL(kernel_state);
  const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  ncclComm_t comm = kernel_state->comm();
  cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
  const std::vector<int64_t>& send_elem_cnts = kernel_state->send_elem_cnts();
  const std::vector<int64_t>& recv_elem_cnts = kernel_state->recv_elem_cnts();
  const int64_t parallel_num = send_elem_cnts.size();
  const DataType data_type = in->data_type();

  std::vector<void*> send_in_ptr;
  std::vector<void*> recv_out_ptr;
  char* buf_ptr = tmp_buffer->mut_dptr<char>();
  int64_t offset = 0;
  for (int64_t i = 0; i < parallel_num; ++i) {
    void* send_ptr = reinterpret_cast<void*>(buf_ptr + offset);
    send_in_ptr.push_back(send_ptr);
    offset += send_elem_cnts.at(i) * GetSizeOfDataType(data_type);
  }
  for (int64_t i = 0; i < parallel_num; ++i) {
    void* recv_ptr = reinterpret_cast<void*>(buf_ptr + offset);
    recv_out_ptr.push_back(recv_ptr);
    offset += recv_elem_cnts.at(i) * GetSizeOfDataType(data_type);
  }

  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec =
      kernel_state->in_tensor_slice_copier_vec();
  for (int64_t i = 0; i < parallel_num; ++i) {
    if (in_tensor_slice_copier_vec.at(i)) {
      in_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), send_in_ptr.at(i), in->dptr());
    }
  }
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  OF_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    if (send_elem_cnts.at(i) != 0) {
      LOG(INFO) << parallel_id << " send " << send_elem_cnts.at(i) << " to " << i;
      OF_NCCL_CHECK(ncclSend(send_in_ptr.at(i), send_elem_cnts.at(i), GetNcclDataType(data_type), i,
                             comm, cuda_stream));
    }
    if (recv_elem_cnts.at(i) != 0) {
      LOG(INFO) << parallel_id << " recv " << recv_elem_cnts.at(i) << " from " << i;
      OF_NCCL_CHECK(ncclRecv(recv_out_ptr.at(i), recv_elem_cnts.at(i), GetNcclDataType(data_type),
                             i, comm, cuda_stream));
    }
  }
  OF_NCCL_CHECK(ncclGroupEnd());
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec =
      kernel_state->out_tensor_slice_copier_vec();

  if (kernel_state->src_nd_sbp_has_no_partial_parallel()) {
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i)) {
        out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(), recv_out_ptr.at(i));
      }
    }
  } else {
    std::unique_ptr<ep::primitive::Add> add_primitive =
        ep::primitive::NewPrimitive<ep::primitive::AddFactory>(ctx->stream()->device_type(),
                                                               out->data_type());
    CHECK(add_primitive);
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
    CHECK(memset_primitive);
    bool is_first_slice = true;
    for (int64_t i = 0; i < parallel_num; ++i) {
      if (out_tensor_slice_copier_vec.at(i)) {
        if (is_first_slice) {
          is_first_slice = false;
          if (recv_elem_cnts.at(i) != out->shape_view().elem_cnt()) {
            // if not same shape, memset out
            memset_primitive->Launch(ctx->stream(), out->mut_dptr(), 0,
                                     out->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
          }
          out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out->mut_dptr(),
                                                  recv_out_ptr.at(i));
        } else {
          if (recv_elem_cnts.at(i) == out->shape_view().elem_cnt()) {
            add_primitive->Launch(ctx->stream(), out->dptr(), recv_out_ptr.at(i), out->mut_dptr(),
                                  out->shape_view().elem_cnt());
          } else {
            void* out_buf = reinterpret_cast<void*>(buf_ptr + offset);
            memset_primitive->Launch(ctx->stream(), out_buf, 0,
                                     out->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
            out_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), out_buf, recv_out_ptr.at(i));
            add_primitive->Launch(ctx->stream(), out->dptr(), out_buf, out->mut_dptr(),
                                  out->shape_view().elem_cnt());
          }
        }
      }
    }
  }
}

size_t InferTmpBufferSize(user_op::InferContext* ctx) {
  const Shape& out_shape = ctx->OutputShape("out", 0);
  const user_op::TensorDesc* logical_in_tensor = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
  const Shape& logical_shape = logical_in_tensor->shape();
  const DataType data_type = logical_in_tensor->data_type();

  const NdSbp& src_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  const NdSbp& dst_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  const int64_t parallel_num = ctx->parallel_num();
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();

  std::vector<TensorSliceView> src_send_intersections;
  std::vector<TensorSliceView> dst_recv_intersections;
  const auto& parallel_desc = ctx->parallel_desc();
  GetRankSendRecvIntersection(parallel_id, /*merge_parallel_desc=*/parallel_desc,
                              /*in_parallel_desc=*/parallel_desc,
                              /*out_parallel_desc=*/parallel_desc, src_nd_sbp, dst_nd_sbp,
                              logical_shape, &src_send_intersections, &dst_recv_intersections);
  int64_t buf_count = 0;
  CHECK_EQ(src_send_intersections.size(), parallel_num);
  for (int64_t i = 0; i < parallel_num; ++i) {
    const TensorSliceView& intersection = src_send_intersections.at(i);
    if (!intersection.IsEmpty()) { buf_count += intersection.shape().elem_cnt(); }
  }
  for (int64_t i = 0; i < parallel_num; ++i) {
    const TensorSliceView& intersection = dst_recv_intersections.at(i);
    if (!intersection.IsEmpty()) { buf_count += intersection.shape().elem_cnt(); }
  }
  if (NdSbpHasPartialParallel(src_nd_sbp)) {
    // Note: when src_nd_sbp has partial_sum, need a out_size buffer to copy and add to out.
    buf_count += out_shape.elem_cnt();
  }
  return buf_count * GetSizeOfDataType(data_type);
}

REGISTER_USER_KERNEL("_nccl_logical_send_recv")
    .SetCreateFn<NcclLogicalSendRecv>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn(InferTmpBufferSize);

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
