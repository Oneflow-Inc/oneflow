#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_info.h"

namespace oneflow {

namespace {

#define DEFINE_STATIC_CASE2KERNEL_MAP(DeviceType, FloatingPointType) \
HashMap<int, std::function<Kernel*()>>& TypeCase2##DeviceType##FloatingPointType##KernelCreator() { \
  static HashMap<int, std::function<Kernel*()>> obj; \
  return obj; \
}

DEFINE_STATIC_CASE2KERNEL_MAP(Cpu, Float);
DEFINE_STATIC_CASE2KERNEL_MAP(Gpu, Float);
DEFINE_STATIC_CASE2KERNEL_MAP(Cpu, Double);
DEFINE_STATIC_CASE2KERNEL_MAP(Gpu, Double);

#undef DEFINE_STATIC_CASE2KERNEL_MAP

Kernel* CreateKernel(OperatorConf::OpTypeCase op_type_case,
                     DeviceType device_type,
                     FloatingPointType floating_point_type) {
  if (device_type == DeviceType::kCPU) {
    if (floating_point_type == FloatingPointType::kFloat) {
      return TypeCase2CpuFloatKernelCreator().at(op_type_case)();
    } else if (floating_point_type == FloatingPointType::kDouble) {
      return TypeCase2CpuDoubleKernelCreator().at(op_type_case)();
    } else {
      LOG(FATAL) << "floating point type has not been set";
      return nullptr;
    }
  } else if (device_type == DeviceType::kGPU) {
    if (floating_point_type == FloatingPointType::kFloat) {
      return TypeCase2GpuFloatKernelCreator().at(op_type_case)();
    } else if (floating_point_type == FloatingPointType::kDouble) {
      return TypeCase2GpuDoubleKernelCreator().at(op_type_case)();
    } else {
      LOG(FATAL) << "floating point type has not been set";
      return nullptr;
    }
  } else {
    LOG(FATAL) << "device type has not been set";
    return nullptr;
  }
}

}  // namespace

void AddCpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
                              std::function<Kernel*()> creator ) {
  CHECK(TypeCase2CpuFloatKernelCreator().emplace(op_type_case, creator).second);
}

void AddGpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
                              std::function<Kernel*()> creator) {
  CHECK(TypeCase2GpuFloatKernelCreator().emplace(op_type_case, creator).second);
}

void AddCpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
                               std::function<Kernel*()> creator) {
  CHECK(TypeCase2CpuDoubleKernelCreator().emplace(op_type_case, creator).second);
}

void AddGpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
                               std::function<Kernel*()> creator) {
  CHECK(TypeCase2GpuDoubleKernelCreator().emplace(op_type_case, creator).second);
}

void KernelMgr::InitFromPlan(const Plan& plan) {
  FloatingPointType floating_point_type = JobDesc::Singleton().floating_point_type();
  uint64_t this_machine_id = RuntimeInfo::Singleton().this_machine_id();
  const PbRpf<std::string>& op_names_rpf =
      plan.machine_id2op_name_set().at(this_machine_id).op_name();
  std::unordered_set<std::string> op_name_set(op_names_rpf.begin(),
                                              op_names_rpf.end());
  for (const OperatorProto& op_proto : plan.op()) {
    const std::string& op_name = op_proto.op_conf().name();
    if (op_name_set.find(op_name) == op_name_set.end()) {
      continue;
    }
    DeviceType device_type = plan.op_name2device_type().at(op_name);
    std::unique_ptr<Kernel> kernel_ptr(CreateKernel(
        op_proto.op_conf().op_type_case(),
        device_type,
        floating_point_type));
    kernel_ptr->InitFromOpProto(op_proto);
    CHECK(op_name2kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
  }
}

}  // namespace oneflow
