#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_user_conf());
  const UserOpConf& user_conf = op.op_conf().user_conf();
  const user_op::OpGradRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpGradRegistryResult(user_conf.op_type_name());
  if (val == nullptr) {
    return Error::GradientFunctionNotFound() << PbMessage2TxtString(op.op_conf());
  }
  user_op::UserOpWrapper user_op(op.op_conf(), LogicalBlobDesc4BnInOp, DiffLbi4BnInOp);
  auto AddOp = [&](const user_op::UserOpConfWrapper& wrapper) {
    op_confs->push_back(wrapper.op_conf());
  };

  val->gen_bw_fn(user_op, AddOp);

  for (const std::string& ibn : op.input_bns()) {
    LogicalBlobId* lbi = DiffLbi4BnInOp(ibn);
    if (lbi != nullptr) {
      CHECK(lbi->has_op_name() && lbi->has_blob_name())
          << " user_op: " << op.op_name() << " op_type_name: " << user_conf.op_type_name()
          << " 's input blob " << ibn << " has not generate input diff blob !";
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kUserConf, &GenerateBackwardOpConf);

}  // namespace oneflow
