#include "kernel/kernel_manager.h"
#include "job/job_desc.h"

namespace oneflow {

namespace {

HashMap<int, std::function<Kernel*()>>& TypeCase2CpuFloatKernelCreator() {
  HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel*()>>& TypeCase2GpuFloatKernelCreator() {
  HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel*()>>& TypeCase2CpuDoubleKernelCreator() {
  HashMap<int, std::function<Kernel*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel*()>>& TypeCase2GpuDoubleKernelCreator() {
  HashMap<int, std::function<Kernel*()>> obj;
  return obj;
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

}  // namespace oneflow
