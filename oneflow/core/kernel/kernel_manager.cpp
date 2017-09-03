#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

HashMap<int, KernelCreator1>& OpCase2Creator() {
  static HashMap<int, KernelCreator1> obj;
  return obj;
}

Kernel* CreateKernel(OperatorConf::OpTypeCase op_case, const OpContext& op_ctx,
                     const OperatorConf& op_conf) {
  return OpCase2Creator().at(op_case)(op_conf, op_ctx);
}

}  // namespace

void AddKernelCreator(OperatorConf::OpTypeCase op_case,
                      KernelCreator1 creator) {
  CHECK(OpCase2Creator().emplace(op_case, creator).second);
}

void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator2) { TODO(); }

void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator3) { TODO(); }
void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator4) { TODO(); }

void KernelMgr::InitFromPlan(const Plan& plan) {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  const PbRpf<std::string>& op_names_rpf =
      plan.machine_id2op_name_set().at(this_machine_id).op_name();
  std::unordered_set<std::string> op_name_set(op_names_rpf.begin(),
                                              op_names_rpf.end());
  for (const OperatorProto& op_proto : plan.op()) {
    const std::string& op_name = op_proto.op_conf().name();
    if (op_name_set.find(op_name) == op_name_set.end()) { continue; }
    OpContext op_ctx = plan.op_name2context().at(op_name);
    LOG(INFO) << "construct kernel: " << op_name;
    std::unique_ptr<Kernel> kernel_ptr(CreateKernel(
        op_proto.op_conf().op_type_case(), op_ctx, op_proto.op_conf()));
    kernel_ptr->InitFromOpProto(op_proto);
    CHECK(op_name2kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
  }
}

}  // namespace oneflow
