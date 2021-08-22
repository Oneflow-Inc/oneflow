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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_xtob_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"

namespace oneflow {

namespace {

Maybe<Symbol<cfg::NdSbp>> GetBroadcastNdSbp() {
  cfg::NdSbp broadcast_nd_sbp;
  broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  return SymbolOf(broadcast_nd_sbp);
}

auto* CachedGetBroadcastNdSbp = DECORATE(&GetBroadcastNdSbp, ThreadLocal);

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

auto* CachedGetBroadcastRoot = DECORATE(&CalBroadcastRoot, ThreadLocal);

Maybe<one::UserOpExpr> EagerNcclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_broadcast", *JUST(UniqueStr("eager_nccl_broadcast")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<int64_t>("root", root)
      .Build();
}

auto* CachedEagerNcclBroadcast = DECORATE(&EagerNcclBroadcast, ThreadLocal);

}  // namespace

Maybe<one::Tensor> NcclXToBBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(out_nd_sbp));
  Symbol<cfg::NdSbp> broadcast_nd_sbp = JUST(CachedGetBroadcastNdSbp());
  const auto& new_tag_in_parallel_desc =
      JUST(ReplaceDeviceType(in_parallel_desc, out_parallel_desc->device_type()));
  std::shared_ptr<one::Tensor> broadcast_input = JUST(one::functional::ToConsistent(
      input, new_tag_in_parallel_desc, *JUST(GetSbpList(broadcast_nd_sbp)), GetNoneSbpList()));
  std::shared_ptr<one::Tensor> local_tensor = JUST(broadcast_input->cur_rank_phy_tensor());
  {
    const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
    if (out_parallel_id->has_value()) {
      const auto& new_in_parallel_id =
          JUST(GetParallelId4CurrentProcessCtx(new_tag_in_parallel_desc));
      if (!new_in_parallel_id->has_value()) {
        std::string device_type = Device::Type4DeviceTag(new_tag_in_parallel_desc->device_tag());
        local_tensor = JUST(one::functional::Empty(*input->shape(), input->dtype(),
                                                   JUST(Device::New(device_type))));
      }
      const auto& broadcast_grop =
          JUST(GetBroadcastGroup(new_tag_in_parallel_desc, out_parallel_desc));

      Symbol<ParallelDesc> broadcast_parallel_desc_cur_rank =
          JUST(MapAt(*broadcast_grop, GlobalProcessCtx::Rank()));
      int64_t root =
          JUST(CachedGetBroadcastRoot(new_tag_in_parallel_desc, broadcast_parallel_desc_cur_rank));
      std::shared_ptr<one::UserOpExpr> op_expr =
          JUST(CachedEagerNcclBroadcast(broadcast_parallel_desc_cur_rank, root));
      local_tensor = JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {local_tensor}));
    }
  }
  return one::functional::ToConsistent(local_tensor, out_parallel_desc,
                                       *JUST(GetSbpList(out_nd_sbp)), GetNoneSbpList());
}

}  // namespace oneflow
