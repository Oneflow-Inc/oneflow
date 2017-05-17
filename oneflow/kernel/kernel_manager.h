#ifndef ONEFLOW_KERNEL_KERNEL_MANAGER_H_
#define ONEFLOW_KERNEL_KERNEL_MANAGER_H_

#include <utility>
#include <memory>
#include <string>
#include "common/util.h"
#include "common/proto_io.h"
#include "kernel/kernel.h"
#include "job/ofelf.pb.h"
#include "job/job_desc.h"

namespace oneflow {

class KernelMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelMgr);
  ~KernelMgr() = default;

  static KernelMgr& Singleton() {
    static KernelMgr obj;
    return obj;
  }

  const Kernel* GetKernelFromOpName(const std::string& op_name) {
    return op_name2kernel_ptr_.at(op_name).get();
  }

  void InitFromELF(const OfElf& of_Elf);

 private:
  KernelMgr() = default;
  HashMap<std::string, std::unique_ptr<const Kernel>> op_name2kernel_ptr_;
};

void AddCpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
                              std::function<Kernel*()> creator);
void AddGpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
                              std::function<Kernel*()> creator);
void AddCpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
                               std::function<Kernel*()> creator);
void AddGpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
                               std::function<Kernel*()> creator);

template<OperatorConf::OpTypeCase op_type_case, typename KernelType>
struct CpuFloatKernelRegister {
  CpuFloatKernelRegister() {
    AddCpuFloatKernelCreator(op_type_case, []() { return new KernelType; });
  }
};

template<OperatorConf::OpTypeCase op_type_case, typename KernelType>
struct GpuFloatKernelRegister {
  GpuFloatKernelRegister() {
    AddGpuFloatKernelCreator(op_type_case, []() { return new KernelType; });
  }
};

template<OperatorConf::OpTypeCase op_type_case, typename KernelType>
struct CpuDoubleKernelRegister {
  CpuDoubleKernelRegister() {
    AddCpuDoubleKernelCreator(op_type_case, []() { return new KernelType; });
  }
};

template<OperatorConf::OpTypeCase op_type_case, typename KernelType>
struct GpuDoubleKernelRegister {
  GpuDoubleKernelRegister() {
    AddGpuDoubleKernelCreator(op_type_case, []() { return new KernelType; });
  }
};

#define REGISTER_KERNEL(OpTypeCase, KernelType) \
  static CpuFloatKernelRegister<OpTypeCase, KernelType<DeviceType::kCPU, FloatingPointType::kFloat>> g_##KernelType##_cpu_float_regst_var; \
  static CpuDoubleKernelRegister<OpTypeCase, KernelType<DeviceType::kCPU, FloatingPointType::kDouble>> g_##KernelType##_cpu_double_regst_var; \
  static GpuFloatKernelRegister<OpTypeCase, KernelType<DeviceType::kGPU, FloatingPointType::kFloat>> g_##KernelType##_gpu_float_regst_var; \
  static GpuDoubleKernelRegister<OpTypeCase, KernelType<DeviceType::kGPU, FloatingPointType::kDouble>> g_##KernelType##_gpu_double_regst_var;

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_MANAGER_H_
