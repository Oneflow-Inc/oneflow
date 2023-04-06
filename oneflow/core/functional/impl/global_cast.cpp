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

#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/common/constant.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

// NOTE: use env variable 'ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE' indicate whether the
// shape and dtype of input tensor on each rank is the same when cast local tensor to global tensor.
// If set true, there will be no meta-information synchronization on each rank.
Optional<bool> ParseEagerLocalToGlobalBalancedOverride() {
  const char* env_p = std::getenv("ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE");
  if (env_p == nullptr) {
    return Optional<bool>();
  } else {
    return ParseBooleanFromEnv("ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE", false);
  }
}

bool NeedSyncAndCheckShapeAndDtype(bool check_meta_hint) {
  thread_local Optional<bool> eager_local_to_global_balanced_override =
      ParseEagerLocalToGlobalBalancedOverride();
  if (eager_local_to_global_balanced_override.has_value()) {
    return IsInDebugMode() || !CHECK_JUST(eager_local_to_global_balanced_override);
  } else {
    return IsInDebugMode() || check_meta_hint;
  }
}

// clang-format off
FLAT_MSG_BEGIN(FlatShapeAndDataType);
  // Methods
  static Maybe<FlatShapeAndDataType> New() {
    const auto& flat_shape_dtype = std::make_shared<FlatShapeAndDataType>();
    flat_shape_dtype->clear();
    return flat_shape_dtype;
  }
  static Maybe<FlatShapeAndDataType> New(const Shape& shape, DataType dtype) {
    const auto& flat_shape_dtype = JUST(New());
    JUST(flat_shape_dtype->mutable_shape()->Init(shape));
    flat_shape_dtype->set_dtype(dtype);
    return flat_shape_dtype;
  }
  Maybe<void> Check(const Shape& shape, DataType dtype) const {
    JUST(this->shape().Check(shape));
    CHECK_EQ_OR_RETURN(this->dtype(), dtype) << Error::RuntimeError()
        << "Expected all tensors on each rank to be the same dtype, but found "
            "at least two dtypes, " << DType(this->dtype()).name() << " and "
        << DType(dtype).name() << "!";
    return Maybe<void>::Ok();
  }
  Maybe<void> Check(const FlatShapeAndDataType& flat_shape_dtype) const {
    JUST(this->shape().Check(flat_shape_dtype.shape()));
    CHECK_EQ_OR_RETURN(this->dtype(), flat_shape_dtype.dtype())
        << Error::RuntimeError()
        << "Expected input of each rank must have the same dtype, but got at least two dtypes, "
        << DType(this->dtype()).name() << " and " << DType(flat_shape_dtype.dtype()).name();
    return Maybe<void>::Ok();
  }
  Maybe<void> ToShape(Shape* shape) const { return this->shape().ToShape(shape); }
  Maybe<Shape> ToShape() const { return shape().ToShape(); }
  int64_t At(int i) const { return shape().At(i); }
  int64_t NumAxes() const { return shape().NumAxes(); }

 private:
  // Fields
  FLAT_MSG_DEFINE_OPTIONAL(FlatShape, shape);
  FLAT_MSG_DEFINE_OPTIONAL(DataType, dtype);
FLAT_MSG_END(FlatShapeAndDataType);
// clang-format on

Maybe<void> ShapeAndDataTypeConsistencyCheck(const Symbol<ParallelDesc>& placement,
                                             const Shape& shape, DataType dtype) {
  if (!placement->containing_current_rank() || placement->parallel_num() == 1) {
    return Maybe<void>::Ok();
  }

  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncLocalShapeDtype));
  const auto& send_buffer = JUST(FlatShapeAndDataType::New(shape, dtype));
  const auto& recv_buffer = JUST(FlatShapeAndDataType::New());
  recv_buffer->clear();

  NaiveAsyncTransportCtx ctx(
      transport_token,
      [send_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = send_buffer.get();
        *size = sizeof(FlatShapeAndDataType);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      },
      [recv_buffer](int64_t rank, void** buffer, std::size_t* size,
                    std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_buffer.get();
        *size = sizeof(FlatShapeAndDataType);
        *Cb = [recv_buffer] {};
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroup::New(placement));
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST_MSG(ctx.WaitDone(), kAsymmetricCodeErrorMsg);
  JUST(send_buffer->Check(*recv_buffer));
  return Maybe<void>::Ok();
}

Maybe<HashMap<int64_t, std::shared_ptr<FlatShapeAndDataType>>> BroadcastGatherShapeAndDataType(
    const Shape& shape, DataType dtype, Symbol<ParallelDesc> parallel_desc) {
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncLocalShapeDtype));
  const auto& send_buffer = JUST(FlatShapeAndDataType::New(shape, dtype));
  const auto& map = std::make_shared<HashMap<int64_t, std::shared_ptr<FlatShapeAndDataType>>>();
  map->emplace(GlobalProcessCtx::Rank(), send_buffer);
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [send_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = send_buffer.get();
        *size = sizeof(FlatShapeAndDataType);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      },
      [map](int64_t rank, void** buffer, std::size_t* size,
            std::function<void()>* Cb) -> Maybe<void> {
        const auto& recv_buffer = JUST(FlatShapeAndDataType::New());
        recv_buffer->clear();
        *buffer = recv_buffer.get();
        *size = sizeof(FlatShapeAndDataType);
        *Cb = [recv_buffer] {};
        CHECK_OR_RETURN(map->emplace(rank, recv_buffer).second);  // NOLINT(maybe-need-error-msg)
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroup::New(parallel_desc));
  JUST(TransportUtil::BroadcastToOtherRanks(rank_group, rank_group, transport_token, &ctx));
  JUST(TransportUtil::CollectFromOtherRanks(rank_group, rank_group, transport_token, &ctx));
  JUST_MSG(ctx.WaitDone(), kAsymmetricCodeErrorMsg);
  return map;
}

Maybe<int64_t> FindRoot(Symbol<ParallelDesc> broadcast_parallel_desc,
                        Symbol<ParallelDesc> src_parallel_desc) {
  for (int64_t process_id : broadcast_parallel_desc->sorted_machine_ids()) {
    if (src_parallel_desc->ContainingMachineId(process_id)) { return process_id; }
  }
  UNIMPLEMENTED_THEN_RETURN();
}

auto* CachedFindRoot = DECORATE(&FindRoot, ThreadLocal);

Maybe<FlatShapeAndDataType> BroadcastShapeAndDtype(const Shape& shape, DataType dtype,
                                                   Symbol<ParallelDesc> parallel_desc) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& rank_group_parallel_desc =
      JUST(RankGroup::GetDefaultParallelDesc(parallel_desc->device_type(), rank_group));
  const auto& process_id2broadcast_group =
      JUST(GetBroadcastGroup(parallel_desc, rank_group_parallel_desc));
  const auto& broadcast_parallel_desc =
      JUST(MapAt(*process_id2broadcast_group, GlobalProcessCtx::Rank()));

  const auto& in_flat_shape_dtype = JUST(FlatShapeAndDataType::New(shape, dtype));
  const auto& out_flat_shape_dtype = JUST(FlatShapeAndDataType::New());
  int64_t root = JUST(CachedFindRoot(broadcast_parallel_desc, parallel_desc));
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncLocalShapeDtype));
  JUST(ccl::CpuBroadcast(in_flat_shape_dtype.get(), out_flat_shape_dtype.get(),
                         sizeof(FlatShapeAndDataType), root, broadcast_parallel_desc,
                         transport_token));
  return out_flat_shape_dtype;
}

Maybe<void> GetConcatenatedShapeAndCheckDtype(
    Shape* logical_shape, DataType* dtype,
    const HashMap<int64_t, std::shared_ptr<FlatShapeAndDataType>>& rank2flat_shape_dtype,
    Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp) {
  *dtype = rank2flat_shape_dtype.begin()->second->dtype();
  HashMap<int64_t, std::shared_ptr<Shape>> rank2logical_shape;
  for (const auto& pair : rank2flat_shape_dtype) {
    rank2logical_shape.emplace(pair.first, JUST(pair.second->ToShape()));
    CHECK_EQ_OR_RETURN(*dtype, pair.second->dtype())
        << Error::RuntimeError()
        << "Expected all tensors on each rank to be the same dtype, but found "
           "at least two dtypes, "
        << DType(*dtype).name() << "(rank " << rank2flat_shape_dtype.begin()->first << ") and "
        << DType(pair.second->dtype()).name() << "(rank " << pair.first << ")!";
  }
  const auto& GetRankPhyShapeByParallelId = [&](Symbol<ParallelDesc> parallel_desc,
                                                int64_t parallel_id) -> Maybe<Shape> {
    int64_t machine_id = JUST(parallel_desc->MachineId4ParallelId(parallel_id));
    return JUST(MapAt(rank2logical_shape, machine_id));
  };
  const auto& parallel_hierarchy = parallel_desc->hierarchy();
  Stride parallel_stride(*parallel_hierarchy);
  for (int32_t i = nd_sbp->sbp_parallel_size() - 1; i >= 0; --i) {
    if (nd_sbp->sbp_parallel(i).has_split_parallel()) {
      int64_t concat_axis = nd_sbp->sbp_parallel(i).split_parallel().axis();
      int64_t group_size = parallel_hierarchy->Count(0, i);
      int64_t stride = parallel_stride.at(i);
      for (int group_id = 0; group_id < group_size; ++group_id) {
        int64_t parallel_num_in_group = parallel_hierarchy->At(i);
        for (int64_t stride_id = 0; stride_id < stride; ++stride_id) {
          ParallelConf parallel_conf;
          parallel_conf.set_device_tag(parallel_desc->device_tag());
          int64_t start_parallel_id = group_id * parallel_num_in_group + stride_id;
          for (int64_t parallel_id_in_group = 0; parallel_id_in_group < parallel_num_in_group;
               ++parallel_id_in_group) {
            int64_t id = start_parallel_id + parallel_id_in_group * stride;
            int64_t machine_id = JUST(parallel_desc->MachineId4ParallelId(id));
            int64_t device_id = JUST(parallel_desc->DeviceId4ParallelId(id));
            parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":"
                                          + std::to_string(device_id));
          }
          Symbol<ParallelDesc> sub_parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
          std::shared_ptr<Shape> first_shape =
              JUST(GetRankPhyShapeByParallelId(sub_parallel_desc, 0));
          CHECK_GE_OR_RETURN(concat_axis, 0)
              << Error::RuntimeError() << "Split axis must not be negative, but got " << concat_axis
              << "!";
          CHECK_LT_OR_RETURN(concat_axis, first_shape->NumAxes())
              << Error::RuntimeError() << "Split axis out of range (expected to be in range of ["
              << 0 << ", " << first_shape->NumAxes() << "), but got " << concat_axis << "!)";

          int64_t logical_concat_dim = first_shape->At(concat_axis);
          for (int parallel_id = 1; parallel_id < sub_parallel_desc->parallel_num();
               ++parallel_id) {
            const auto& rank_shape =
                JUST(GetRankPhyShapeByParallelId(sub_parallel_desc, parallel_id));
            CHECK_EQ_OR_RETURN(rank_shape->NumAxes(), first_shape->NumAxes())
                << Error::RuntimeError() << "Sizes of tensors must match except in dimension "
                << concat_axis << ", but found " << first_shape->ToString() << "(rank "
                << JUST(sub_parallel_desc->MachineId4ParallelId(0)) << ") and "
                << rank_shape->ToString() << "(rank "
                << JUST(sub_parallel_desc->MachineId4ParallelId(parallel_id)) << ")!";
            logical_concat_dim += rank_shape->At(concat_axis);
          }

          BalancedSplitter bs(logical_concat_dim, sub_parallel_desc->parallel_num());
          CHECK_EQ_OR_RETURN(first_shape->At(concat_axis), bs.At(0).size())
              << Error::RuntimeError() << "Sizes of tensors in dimension " << concat_axis
              << " must be same or match balanced split distribution. See "
                 "https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/common/"
                 "balanced_splitter.h "
                 "for details of balanced split";
          first_shape->Set(concat_axis, logical_concat_dim);

          for (int parallel_id = 1; parallel_id < sub_parallel_desc->parallel_num();
               ++parallel_id) {
            std::shared_ptr<Shape> rank_shape =
                JUST(GetRankPhyShapeByParallelId(sub_parallel_desc, parallel_id));
            for (int i = 0; i < first_shape->NumAxes(); ++i) {
              if (i == concat_axis) {
                CHECK_EQ_OR_RETURN(rank_shape->At(i), bs.At(parallel_id).size())
                    << Error::RuntimeError() << "Sizes of tensors in dimension " << concat_axis
                    << " must be same or match balanced split distribution. See "
                       "https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/common/"
                       "balanced_splitter.h "
                       "for details of balanced split";
              } else {
                CHECK_EQ_OR_RETURN(rank_shape->At(i), first_shape->At(i))
                    << Error::RuntimeError() << "Sizes of tensors must match except in dimension "
                    << concat_axis << ". Expected size " << first_shape->At(i) << " but got size "
                    << rank_shape->At(i) << " for tensor on rank "
                    << JUST(sub_parallel_desc->MachineId4ParallelId(parallel_id)) << "!";
              }
            }
            rank_shape->Set(concat_axis, logical_concat_dim);
          }
        }
      }
    }
  }
  *logical_shape = *JUST(GetRankPhyShapeByParallelId(parallel_desc, 0));
  return Maybe<void>::Ok();
}

Maybe<void> GetLogicalShapeAndDataType(Shape* logical_shape, DataType* /* in and out */ dtype,
                                       std::shared_ptr<const Shape> physical_shape,
                                       Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp,
                                       bool sync_and_check_meta) {
  if (!sync_and_check_meta) {
    *logical_shape = *JUST(GetLogicalShape(*physical_shape, *nd_sbp, *parallel_desc));
  } else {
    if (ContainSplitSbp(nd_sbp)) {
      *logical_shape = *physical_shape;
      if (parallel_desc->containing_current_rank()) {
        const auto& rank2flat_shape_dtype =
            JUST(BroadcastGatherShapeAndDataType(*logical_shape, *dtype, parallel_desc));
        JUST(GetConcatenatedShapeAndCheckDtype(logical_shape, dtype, *rank2flat_shape_dtype,
                                               parallel_desc, nd_sbp));
      }
    } else {
      *logical_shape = *physical_shape;
      JUST(ShapeAndDataTypeConsistencyCheck(parallel_desc, *logical_shape, *dtype));
    }
  }
  if (JUST(RankGroup::New(parallel_desc)) != JUST(RankGroupScope::CurrentRankGroup())) {
    const auto& flat_shape_dtype =
        JUST(BroadcastShapeAndDtype(*logical_shape, *dtype, parallel_desc));
    *logical_shape = *JUST(flat_shape_dtype->ToShape());
    *dtype = flat_shape_dtype->dtype();
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckNdSbpValid(Symbol<NdSbp> nd_sbp, const Shape& logical_shape) {
  for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    const auto& sbp_parallel = nd_sbp->sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      CHECK_LT_OR_RETURN(sbp_parallel.split_parallel().axis(), logical_shape.NumAxes())
          << Error::RuntimeError() << "Split axis out of range (expected to be in range of [" << 0
          << ", " << logical_shape.NumAxes() << "), but got "
          << sbp_parallel.split_parallel().axis() << "!)";
    }
  }
  return Maybe<void>::Ok();
}

namespace {

Maybe<one::OpExpr> RawGetGlobalToGlobalOpExpr(
    const std::vector<Symbol<SbpParallel>>& grad_sbp_parallels) {
  Optional<Symbol<NdSbp>> grad_nd_sbp;
  if (!grad_sbp_parallels.empty()) { grad_nd_sbp = JUST(GetNdSbp(grad_sbp_parallels)); }
  std::shared_ptr<one::OpExpr> op_expr = JUST(one::GlobalToGlobalOpExpr::New(grad_nd_sbp));
  return op_expr;
}

}  // namespace

static constexpr auto* GetGlobalToGlobalOpExpr =
    DECORATE(&RawGetGlobalToGlobalOpExpr, ThreadLocalCopiable);

Maybe<Tensor> GlobalToGlobal(const std::shared_ptr<Tensor>& x, Symbol<ParallelDesc> parallel_desc,
                             const std::vector<Symbol<SbpParallel>>& sbp_parallels,
                             const std::vector<Symbol<SbpParallel>>& grad_sbp_parallels,
                             bool copy) {
  const auto& global_tensor = JUST(x->AsGlobalTensor());
  CHECK_NOTNULL_OR_RETURN(global_tensor) << "global tensors supported only";
  const auto& nd_sbp = JUST(GetNdSbp(sbp_parallels));
  JUST(CheckNdSbpValid(nd_sbp, *x->shape()));
  std::shared_ptr<one::OpExpr> op;
  if (unlikely(!LazyMode::is_enabled()
               && JUST(x->parallel_desc())->hierarchy()->NumAxes()
                      != parallel_desc->hierarchy()->NumAxes()
               && grad_sbp_parallels.size() == 0)) {
    op = JUST(GetGlobalToGlobalOpExpr(*JUST(GetSbpList(JUST(x->nd_sbp())))));
  } else {
    op = JUST(GetGlobalToGlobalOpExpr(grad_sbp_parallels));
  }
  if (!LazyMode::is_enabled() && JUST(x->nd_sbp()) == nd_sbp
      && JUST(x->parallel_desc()) == parallel_desc && grad_sbp_parallels.size() == 0) {
    if (copy) { return functional::Identity(x); }
    return x;
  }
  const auto& tensor = JUST(OpInterpUtil::Dispatch<one::Tensor>(
      *op, {global_tensor}, OpExprInterpContext(AttrMap{}, parallel_desc, nd_sbp)));
  if (!LazyMode::is_enabled() && tensor != x && !IsGlobalTensorMetaCheckDisabled()) {
    const auto& input_global_id = JUST(x->transport_token());
    const auto& output_consistend_id = JUST(tensor->transport_token());
    CHECK_NE_OR_RETURN(input_global_id, output_consistend_id);  // NOLINT(maybe-need-error-msg)
  }
  return tensor;
}

Maybe<Tensor> LocalToGlobal(const std::shared_ptr<Tensor>& x, Symbol<ParallelDesc> parallel_desc,
                            const std::vector<Symbol<SbpParallel>>& sbp_parallels,
                            const std::shared_ptr<OpExpr>& op, bool check_meta_hint, bool copy) {
  CHECK_OR_RETURN(!x->is_lazy())
      << Error::RuntimeError()
      << "local_tensor.to_global() is not supported within nn.Graph for now";
  CHECK_OR_RETURN(x->is_local()) << Error::RuntimeError() << "local tensors supported only";
  std::shared_ptr<one::Tensor> input = x;
  // copy to right device first if input's device type is wrong
  if (JUST(input->device())->type() != parallel_desc->device_tag()) {
    VLOG(2) << "The device_type of the input tensor is different from placement, now copy it to "
            << parallel_desc->device_tag();
    input = JUST(functional::Copy(x, parallel_desc->device_tag(), GlobalProcessCtx::LocalRank(),
                                  /*pin_memory=*/false));
  }
  // copy to default device of the current rank if input's device type is right but not on default
  // device
  bool device_mismatch = JUST(input->device())->device_id() != GlobalProcessCtx::LocalRank();
  if (copy || device_mismatch) {
    if (device_mismatch) {
      VLOG(2) << "The tensor isn't on default device of the current rank, now copy it to "
              << parallel_desc->device_tag() << ": " << GlobalProcessCtx::LocalRank();
    }
    input = JUST(functional::Copy(x, parallel_desc->device_tag(), GlobalProcessCtx::LocalRank(),
                                  /*pin_memory=*/false));
  }
  const auto& device = JUST(input->device());
  CHECK_EQ_OR_RETURN(device->type(), parallel_desc->device_tag())
      << Error::UnimplementedError() << "tensor' device type must be same with placement.";
  CHECK_EQ_OR_RETURN(device->device_id(), GlobalProcessCtx::LocalRank())
      << Error::UnimplementedError() << "tensor must be on default device of the current rank.";
  Symbol<NdSbp> nd_sbp = JUST(GetNdSbp(sbp_parallels));
  const auto& shape = std::make_shared<Shape>();
  DataType dtype = x->dtype()->data_type();
  bool sync_and_check_meta = NeedSyncAndCheckShapeAndDtype(check_meta_hint);
  JUST(GetLogicalShapeAndDataType(shape.get(), &dtype, x->shape(), parallel_desc, nd_sbp,
                                  sync_and_check_meta));
  auto& attrs =
      THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "sync_data", "inplace_when_sync_data");
  attrs.SetAllAttrs(*shape, dtype, true, !copy);
  const auto& output = JUST(OpInterpUtil::Dispatch<one::Tensor>(
      *op, {input}, OpExprInterpContext(attrs, parallel_desc, nd_sbp)));
  return output;
}

}  //  namespace

class LocalToGlobalFunctor {
 public:
  LocalToGlobalFunctor() {
    op_ = CHECK_JUST(one::LocalToGlobalOpExpr::New(*CHECK_JUST(UniqueStr("local_to_global"))));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& sbp_parallels,
                           const Shape& shape, const Symbol<DType>& dtype, bool sync_data,
                           bool copy) const {
    JUST(CheckDeviceIdsIsValid(parallel_desc));
    NonRecursiveMetaInfoConsistencyCheckScope no_recursive_meta_info_conisitency_check_scope;
    JUST(MetaInfoConsistencyCheck(parallel_desc, sbp_parallels, 1, /* force_check */ false));
    CHECK_OR_RETURN(x->is_local())
        << Error::RuntimeError()
        << "Expected local tensor for local_to_global but got global tensor!";
    std::shared_ptr<one::Tensor> input = x->contiguous();
    // copy to right device first if input's device type is wrong
    if (JUST(input->device())->type() != parallel_desc->device_tag()) {
      VLOG(2) << "The device_type of the input tensor is different from placement, now copy it to "
              << parallel_desc->device_tag();
      input = JUST(functional::Copy(x, parallel_desc->device_tag(), GlobalProcessCtx::LocalRank(),
                                    /*pin_memory=*/false));
    }
    // copy to default device of the current rank if input's device type is right but not on default
    // device
    bool device_mismatch = JUST(input->device())->device_id() != GlobalProcessCtx::LocalRank();
    if (copy || device_mismatch) {
      if (device_mismatch) {
        VLOG(2) << "The tensor isn't on default device of the current rank, now copy it to "
                << parallel_desc->device_tag() << ": " << GlobalProcessCtx::LocalRank();
      }
      input = JUST(functional::Copy(x, parallel_desc->device_tag(), GlobalProcessCtx::LocalRank(),
                                    /*pin_memory=*/false));
    }
    Symbol<NdSbp> nd_sbp = JUST(GetNdSbp(sbp_parallels));
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "sync_data", "inplace_when_sync_data");
    attrs.SetAllAttrs(shape, dtype->data_type(), sync_data, !copy);
    DisableCheckGlobalTensorMetaScope scope{};
    const auto& tensor = JUST(OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {input}, OpExprInterpContext(attrs, parallel_desc, nd_sbp)));
    return tensor;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ToGlobalFunctor {
 public:
  ToGlobalFunctor() {
    local_to_global_op_ =
        CHECK_JUST(one::LocalToGlobalOpExpr::New(*CHECK_JUST(UniqueStr("local_to_global"))));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> parallel_desc,
                           const std::vector<Symbol<SbpParallel>>& sbp_parallels,
                           const std::vector<Symbol<SbpParallel>>& grad_sbp_parallels,
                           bool check_meta, bool copy) const {
    JUST(CheckDeviceIdsIsValid(parallel_desc));
    NonRecursiveMetaInfoConsistencyCheckScope scope;
    JUST(MetaInfoConsistencyCheck(parallel_desc, sbp_parallels, grad_sbp_parallels, 1,
                                  /* force_check */ check_meta));
    std::shared_ptr<Tensor> tensor;
    if (x->is_global()) {
      tensor = JUST(GlobalToGlobal(x, parallel_desc, sbp_parallels, grad_sbp_parallels, copy));
    } else {
      DeviceType device_type = parallel_desc->device_type();
      if (device_type == DeviceType::kCPU || device_type == DeviceType::kCUDA) {
        tensor = JUST(
            LocalToGlobal(x, parallel_desc, sbp_parallels, local_to_global_op_, check_meta, copy));
      } else {
        // Assuming that the newly adapted hardware device does not support collective
        // communication, since local to global may need to synchronize data (through the
        // broadcast API), if device_type is neither cpu nor cuda, generate global tensor
        // with the corresponding cpu placement first, then convert the cpu global tensor
        // to the desired placement.
        Symbol<ParallelDesc> cpu_parallel_desc =
            JUST(ReplaceDeviceType(parallel_desc, DeviceType::kCPU));
        std::shared_ptr<Tensor> cpu_tensor = JUST(LocalToGlobal(
            x, cpu_parallel_desc, sbp_parallels, local_to_global_op_, check_meta, copy));
        tensor =
            JUST(GlobalToGlobal(cpu_tensor, parallel_desc, sbp_parallels, GetNoneSbpList(), copy));
      }
    }
    return tensor;
  }

 private:
  std::shared_ptr<OpExpr> local_to_global_op_;
};

class GlobalToLocalFunctor {
 public:
  GlobalToLocalFunctor() {
    op_ = CHECK_JUST(one::GlobalToLocalOpExpr::New(*CHECK_JUST(UniqueStr("global_to_local"))));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, bool copy) const {
    CHECK_OR_RETURN(!x->is_lazy())
        << Error::RuntimeError()
        << "global_tensor.to_local() is not supported within nn.Graph for now";
    CHECK_OR_RETURN(x->is_global())
        << Error::RuntimeError() << "Expected global tensor for to_local but got local tensor!";
    const auto& local_tensor = JUST(OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}));
    if (copy) { return local_tensor->clone(); }
    return local_tensor;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::LocalToGlobalFunctor>("LocalToGlobal");
  m.add_functor<impl::ToGlobalFunctor>("ToGlobal");
  m.add_functor<impl::GlobalToLocalFunctor>("GlobalToLocal");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
