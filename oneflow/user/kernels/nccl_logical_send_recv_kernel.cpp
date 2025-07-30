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
#include "collective_communication/include/collective_communication.h"
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
#include "oneflow/user/kernels/collective_communication/include/all_to_all.h"

#if (defined(WITH_CUDA) && (NCCL_VERSION_CODE > 2700)) || defined(WITH_NPU) || defined(WITH_MLU) || defined(WITH_XPU)

namespace oneflow {

class CclLogicalSendRecvState final : public user_op::OpKernelState {
 public:
  explicit CclLogicalSendRecvState(user_op::KernelInitContext* ctx);
  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec() const {
    return in_tensor_slice_copier_vec_;
  }
  const std::vector<std::shared_ptr<TensorSliceCopier>>& out_tensor_slice_copier_vec() const {
    return out_tensor_slice_copier_vec_;
  }
  bool src_nd_sbp_has_no_partial_parallel() const { return src_nd_sbp_no_partial_parallel_; }
  const std::vector<int64_t>& send_elem_cnts() const { return send_elem_cnts_; }
  const std::vector<int64_t>& recv_elem_cnts() const { return recv_elem_cnts_; }
  ccl::CclComm ccl_comm() const { return GetOrCreateComm().ccl_comm; }

 private:
  struct Comm {
    Comm(ccl::CclComm comm) : ccl_comm(comm) {}
    ccl::CclComm ccl_comm;
  };

  void InitComm() const;
  const Comm& GetOrCreateComm() const {
    if (!ccl_comm_) { InitComm(); }
    return *ccl_comm_;
  }

  std::string stream_name_;
  std::unique_ptr<ParallelDesc> parallel_desc_;
  mutable std::unique_ptr<Comm> ccl_comm_;
  bool src_nd_sbp_no_partial_parallel_;
  std::vector<std::shared_ptr<TensorSliceCopier>> in_tensor_slice_copier_vec_;
  std::vector<std::shared_ptr<TensorSliceCopier>> out_tensor_slice_copier_vec_;
  std::vector<int64_t> send_elem_cnts_;
  std::vector<int64_t> recv_elem_cnts_;
};

CclLogicalSendRecvState::CclLogicalSendRecvState(user_op::KernelInitContext* ctx)
    : stream_name_(EagerCclCommMgr::kDefaultCclStreamName) {
  if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  parallel_desc_ = std::make_unique<ParallelDesc>(ctx->parallel_desc());
  NdSbp src_nd_sbp;
  CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", &src_nd_sbp));
  NdSbp dst_nd_sbp;
  CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", &dst_nd_sbp));
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

void CclLogicalSendRecvState::InitComm() const {
  EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
  ccl::CclComm ccl_comm =
      comm_mgr->GetCclCommForParallelDescAndStreamName(*parallel_desc_.get(), stream_name_);
  ccl_comm_.reset(new Comm(ccl_comm));
}

class CclLogicalSendRecv final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclLogicalSendRecv);
  CclLogicalSendRecv() = default;
  ~CclLogicalSendRecv() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<CclLogicalSendRecvState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

void CclLogicalSendRecv::Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                                 const user_op::OpKernelCache*) const {
  auto* kernel_state = dynamic_cast<CclLogicalSendRecvState*>(state);
  CHECK_NOTNULL(kernel_state);
  const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  ccl::CclComm ccl_comm = kernel_state->ccl_comm();
  const std::vector<int64_t>& send_elem_cnts = kernel_state->send_elem_cnts();
  const std::vector<int64_t>& recv_elem_cnts = kernel_state->recv_elem_cnts();
  const int64_t parallel_num = send_elem_cnts.size();
  const DataType data_type = in->data_type();

  std::vector<void*> send_in_ptr;
  std::vector<void*> recv_out_ptr;
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> recv_offsets;
  char* buf_ptr = tmp_buffer->mut_dptr<char>();
  uint64_t offset = 0;
  for (int64_t i = 0; i < parallel_num; ++i) {
    void* send_ptr = reinterpret_cast<void*>(buf_ptr + offset);
    send_in_ptr.push_back(send_ptr);
    send_offsets.push_back(offset);
    offset += send_elem_cnts.at(i) * GetSizeOfDataType(data_type);
  }
  const uint64_t recv_offset = offset;
  for (int64_t i = 0; i < parallel_num; ++i) {
    void* recv_ptr = reinterpret_cast<void*>(buf_ptr + offset);
    recv_out_ptr.push_back(recv_ptr);
    recv_offsets.push_back(offset - recv_offset);
    offset += recv_elem_cnts.at(i) * GetSizeOfDataType(data_type);
  }

  const std::vector<std::shared_ptr<TensorSliceCopier>>& in_tensor_slice_copier_vec =
      kernel_state->in_tensor_slice_copier_vec();
  for (int64_t i = 0; i < parallel_num; ++i) {
    if (in_tensor_slice_copier_vec.at(i)) {
      in_tensor_slice_copier_vec.at(i)->Copy(ctx->stream(), send_in_ptr.at(i), in->dptr());
    }
  }

  std::unique_ptr<ccl::AllToAll> all_to_all = ccl::NewCollectiveCommunication<ccl::AllToAll>(
      ctx->stream()->device_type(), data_type, data_type, parallel_num);
  void* send_buf = reinterpret_cast<void*>(buf_ptr);
  void* recv_buf = reinterpret_cast<void*>(buf_ptr + recv_offset);
  all_to_all->Launch(ctx->stream(), send_buf, send_elem_cnts.data(), send_offsets.data(), recv_buf,
                     recv_elem_cnts.data(), recv_offsets.data(), ccl_comm, /*has_input=*/true,
                     /*has_output=*/true);

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

// TODO:(zhaoluyang) SetIsMatchedHob support multi devices(not including cpu)
REGISTER_USER_KERNEL("_nccl_logical_send_recv")
    .SetCreateFn<CclLogicalSendRecv>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     || (user_op::HobDeviceType() == DeviceType::kNPU)
                     || (user_op::HobDeviceType() == DeviceType::kMLU))
    .SetInferTmpSizeFn(InferTmpBufferSize);

}  // namespace oneflow

#endif  // WITH_CUDA || WITH_NPU || WITH_MLU
