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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_xtob_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_nto1_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_1ton_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"

namespace oneflow {

namespace {

Maybe<Symbol<ParallelDesc>> GetOverlapParallelDescWithParallelNumEqualsOne(
    Symbol<ParallelDesc> lhs_parallel_desc, Symbol<ParallelDesc> rhs_parallel_desc) {
  CHECK_EQ_OR_RETURN(lhs_parallel_desc->device_tag(), rhs_parallel_desc->device_tag());
  int64_t machine_id = -1;
  int64_t device_id = -1;
  for (int64_t mach_id : lhs_parallel_desc->sorted_machine_ids()) {
    bool machine_and_device_id_inited = false;
    for (int64_t dev_id : lhs_parallel_desc->sorted_dev_phy_ids(mach_id)) {
      if (rhs_parallel_desc->Containing(mach_id, dev_id)) {
        machine_id = mach_id;
        device_id = dev_id;
        machine_and_device_id_inited = true;
        break;
      }
    }
    if (machine_and_device_id_inited) { break; }
  }
  std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
  parallel_conf->set_device_tag(lhs_parallel_desc->device_tag());
  parallel_conf->add_device_name("@" + std::to_string(machine_id) + ":"
                                 + std::to_string(device_id));
  std::shared_ptr<ParallelDesc> parallel_desc;
  JUST(LogicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
    parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
    return Maybe<void>::Ok();
  }));
  return SymbolOf(*parallel_desc);
}

auto* CachedGetOverlapParallelDescWithParallelNumEqualsOne =
    DECORATE(&GetOverlapParallelDescWithParallelNumEqualsOne, ThreadLocal);

}  // namespace

Maybe<one::Tensor> NcclXToBBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(out_nd_sbp));
  CHECK_EQ_OR_RETURN(in_parallel_desc->device_tag(), "gpu");
  CHECK_EQ_OR_RETURN(out_parallel_desc->device_tag(), "gpu");
  Symbol<ParallelDesc> parallel_desc_with_parallel_num_eq_one = JUST(
      CachedGetOverlapParallelDescWithParallelNumEqualsOne(in_parallel_desc, out_parallel_desc));
  // n -> 1, 1 -> n
  std::shared_ptr<one::Tensor> mid_tensor =
      JUST(one::functional::ToConsistent(input, parallel_desc_with_parallel_num_eq_one,
                                         *JUST(GetSbpList(in_nd_sbp)), GetNoneSbpList()));
  return one::functional::ToConsistent(mid_tensor, out_parallel_desc, *JUST(GetSbpList(out_nd_sbp)),
                                       GetNoneSbpList());
}

}  // namespace oneflow
