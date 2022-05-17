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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {
bool IsAllBroadcastNdSbp(Symbol<NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_broadcast_parallel()) { return false; }
  }
  return true;
}

Maybe<void> RawCheckAsymmetricBroadcast(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                        const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(out->nd_sbp()));
  CHECK_OR_RETURN(out->placement()->Bigger(*in->placement())
                  || in->placement()->Bigger(*out->placement()));
  return Maybe<void>::Ok();
}

static constexpr auto* CheckAsymmetricBroadcast =
    DECORATE(&RawCheckAsymmetricBroadcast, ThreadLocalCachedCopiable);

Maybe<int64_t> CalBroadcastRoot(Symbol<ParallelDesc> src_parallel_desc,
                                Symbol<ParallelDesc> dst_parallel_desc) {
  int64_t machine_id = -1;
  int64_t device_id = -1;
  for (int64_t mach_id : src_parallel_desc->sorted_machine_ids()) {
    bool machine_and_device_id_inited = false;
    for (int64_t dev_id : src_parallel_desc->sorted_dev_phy_ids(mach_id)) {
      if (dst_parallel_desc->Containing(mach_id, dev_id)) {
        machine_id = mach_id;
        device_id = dev_id;
        machine_and_device_id_inited = true;
        break;
      }
    }
    if (machine_and_device_id_inited) { break; }
  }
  CHECK_OR_RETURN(machine_id != -1 && device_id != -1);
  return machine_id;
}

static constexpr auto* CachedGetBroadcastRoot = DECORATE(&CalBroadcastRoot, ThreadLocalCached);

Maybe<one::UserOpExpr> EagerNcclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_broadcast", *JUST(UniqueStr("eager_nccl_broadcast")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<int64_t>("root", root)
      .Build();
}

static constexpr auto* CachedEagerNcclBroadcast = DECORATE(&EagerNcclBroadcast, ThreadLocalCached);
}  // namespace

Maybe<one::Tensor> AsymmetricBroadcast(const std::shared_ptr<one::Tensor>& tensor,
                                       Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  const auto& in_placement = in->placement();
  const auto& out_placement = out->placement();
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in_placement);
  std::shared_ptr<one::Tensor> local_tensor = JUST(tensor->cur_rank_phy_tensor());
  if (out->placement()->Bigger(*in->placement())) {
    const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_placement));
    if (out_parallel_id->has_value()) {
      const auto& in_parallel_id = JUST(GetParallelId4CurrentProcessCtx(in_placement));
      if (!in_parallel_id->has_value()) {
        const std::string& device_type = in_placement->device_tag();
        local_tensor =
            JUST(one::functional::Empty(*tensor->shape(), tensor->dtype(),
                                        JUST(Device::New(device_type)), /*pin_memory=*/false));
      }
      const auto& broadcast_group = JUST(GetBroadcastGroup(in_placement, out_placement));

      Symbol<ParallelDesc> broadcast_placement_cur_rank =
          JUST(MapAt(*broadcast_group, GlobalProcessCtx::Rank()));
      int64_t root = JUST(CachedGetBroadcastRoot(in_placement, broadcast_placement_cur_rank));
      std::shared_ptr<one::UserOpExpr> op_expr =
          JUST(CachedEagerNcclBroadcast(broadcast_placement_cur_rank, root));
      local_tensor = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {local_tensor}));
    }
  }
  return one::functional::LocalToConsistent(local_tensor, out_placement,
                                            *JUST(GetSbpList(out->nd_sbp())), *tensor->shape(),
                                            tensor->dtype());
}

COMMAND(RegisterBoxingFunction("asymmetric-broadcast", CheckAsymmetricBroadcast,
                               &AsymmetricBroadcast));

}  // namespace oneflow
