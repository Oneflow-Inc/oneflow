#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

std::string ComputeTag(OperatorConf::OpTypeCase op_case,
                       DeviceType device_type) {
  std::stringstream ss;
  ss << op_case << "," << device_type;
  return ss.str();
}

HashMap<std::string, std::function<Kernel*(const OperatorConf&)>>&
Tag2Creator() {
  static HashMap<std::string, std::function<Kernel*(const OperatorConf&)>> obj;
  return obj;
}

Kernel* CreateKernel(OperatorConf::OpTypeCase op_case, DeviceType device_type,
                     const OperatorConf& op_conf) {
  std::string tag = ComputeTag(op_case, device_type);
  return Tag2Creator().at(tag)(op_conf);
}

}  // namespace

void AddKernelCreator(OperatorConf::OpTypeCase op_case, DeviceType device_type,
                      std::function<Kernel*(const OperatorConf&)> creator) {
  std::string tag = ComputeTag(op_case, device_type);
  CHECK(Tag2Creator().emplace(tag, creator).second);
}

void KernelMgr::InitFromPlan(const Plan& plan) {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  const PbRpf<std::string>& op_names_rpf =
      plan.machine_id2op_name_set().at(this_machine_id).op_name();
  std::unordered_set<std::string> op_name_set(op_names_rpf.begin(),
                                              op_names_rpf.end());
  for (const OperatorProto& op_proto : plan.op()) {
    const std::string& op_name = op_proto.op_conf().name();
    if (op_name_set.find(op_name) == op_name_set.end()) { continue; }
    DeviceType device_type = plan.op_name2device_type().at(op_name);
    LOG(INFO) << "construct kernel: " << op_name;
    std::unique_ptr<Kernel> kernel_ptr(CreateKernel(
        op_proto.op_conf().op_type_case(), device_type, op_proto.op_conf()));
    kernel_ptr->InitFromOpProto(op_proto);
    CHECK(op_name2kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
  }
}

}  // namespace oneflow
