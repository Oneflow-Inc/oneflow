#include "kernel/kernel_manager.h"
#include "job/job_desc.h"

namespace oneflow {

namespace {

HashMap<int, std::function<Kernel<float>*()>>& TypeCase2CpuFloatKernelCreator() {
  HashMap<int, std::function<Kernel<float>*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel<float>*()>>& TypeCase2GpuFLoatKernelCreator() {
  HashMap<int, std::function<Kernel<float>*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel<double>*()>>& TypeCase2CpuDoubleKernelCreator() {
  HashMap<int, std::function<Kernel<double>*()>> obj;
  return obj;
}

HashMap<int, std::function<Kernel<double>*()>>& TypeCase2GpuDoubleKernelCreator() {
  HashMap<int, std::function<Kernel<double>*()>> obj;
  return obj;
}

}

void AddCpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
                         std::function<Kernel<float>*()> creator ) {
  CHECK(TypeCase2CpuFloatKernelCreator().emplace(op_type_case, creator).second);
}

void AddGpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
                         std::function<Kernel<float>*()> creator) {
  CHECK(TypeCase2GpuFLoatKernelCreator().emplace(op_type_case, creator).second);
}

void AddCpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
  std::function<Kernel<double>*()> creator) {
  CHECK(TypeCase2CpuDoubleKernelCreator().emplace(op_type_case, creator).second);
}

void AddGpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
  std::function<Kernel<double>*()> creator) {
  CHECK(TypeCase2GpuDoubleKernelCreator().emplace(op_type_case, creator).second);
}

template<typename Dtype>
Kernel<Dtype>* CreateKernel(OperatorConf::OpTypeCase op_type_case) {
  DeviceType device_type = JobDesc::Singleton().resource().device_type();
  if (device_type == DeviceType::kCPU) {
    return TypeCase2CpuKernelCreator().at(op_type_case)();
  } else if (device_type == DeviceType::kGPU) {
    return TypeCase2GpuKernelCreator().at(op_type_case)();
  } else {
    LOG(FATAL) << "device type has not been set";
    return nullptr;
  }
}

}  // namespace oneflow
