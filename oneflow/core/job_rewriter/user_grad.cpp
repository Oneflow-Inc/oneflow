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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& fw_op, std::vector<OperatorConf>* bw_op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK_OR_RETURN(fw_op.op_conf().has_user_conf());
  const UserOpConf& user_conf = fw_op.op_conf().user_conf();
  const user_op::OpGradRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpGradRegistryResult(user_conf.op_type_name());
  CHECK_NOTNULL_OR_RETURN(val) << Error::GradientFunctionNotFoundError()
                               << " op cannot find backward op in autograd, forward op: "
                               << PbMessage2TxtString(fw_op.op_conf());

  user_op::UserOpWrapper fw_user_op(fw_op.op_conf(), LogicalBlobDesc4BnInOp, DiffLbi4BnInOp);
  if (nullptr != val->bw_gen_fn) {
    // new refined interface
    user_op::BackwardOpConfContext ctx(fw_user_op, bw_op_confs);
    JUST(val->bw_gen_fn(&ctx));
  } else if (nullptr != val->gen_bw_fn) {
    // old interface, will be removed when all backward gradient configs are using new interface
    auto AddOp = [&](const user_op::UserOpConfWrapper& wrapper) {
      bw_op_confs->emplace_back(wrapper.op_conf());
    };
    JUST(val->gen_bw_fn(fw_user_op, AddOp));
  }

  for (const std::string& ibn : fw_op.input_bns()) {
    LogicalBlobId* lbi = DiffLbi4BnInOp(ibn);
    if (lbi != nullptr) {
      CHECK_OR_RETURN(lbi->has_op_name() && lbi->has_blob_name())
          << " user_op: " << fw_op.op_name() << " op_type_name: " << user_conf.op_type_name()
          << " 's input blob " << ibn << " has not generate input diff blob !";
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kUserConf, &GenerateBackwardOpConf);

}  // namespace oneflow
