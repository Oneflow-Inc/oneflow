#include "kernel/kernel_manager.h"
#include "job/job_desc.h"

namespace oneflow {

namespace {

HashMap<int, std::function<Kernel*()>>& TypeCase2CpuFloatKernelCreator() {
  static HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel*()>>& TypeCase2GpuFloatKernelCreator() {
  static HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel*()>>& TypeCase2CpuDoubleKernelCreator() {
  static HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel*()>>& TypeCase2GpuDoubleKernelCreator() {
  static HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

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

void KernelMgr::InitFromELF(const OfElf& of_Elf) {
  const PbRpf<OperatorProto>& op_protos = of_Elf.op();
  FloatingPointType floating_point_type = JobDesc::Singleton().floating_point_type();
  for (const OperatorProto& op_proto : op_protos) {
    const std::string& op_name = op_proto.op_conf().name();
    DeviceType device_type = of_Elf.op_name2device_type().at(op_name);
    std::unique_ptr<Kernel> kernel_ptr(CreateKernel(
      op_proto.op_conf().op_type_case(),
      device_type,
      floating_point_type));
    kernel_ptr->InitFromOpProto(op_proto);
    CHECK(op_name2kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
  }
}

}  // namespace oneflow
