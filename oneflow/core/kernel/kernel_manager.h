#ifndef ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class KernelMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelMgr);
  ~KernelMgr() = default;

  OF_SINGLETON(KernelMgr);

  const Kernel* GetKernelFromOpName(const std::string& op_name) {
    return op_name2kernel_ptr_.at(op_name).get();
  }

  void InitFromPlan(const Plan&);

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

#define REGISTER_CPU_KERNEL(OpTypeCase, KernelType)                    \
  static CpuFloatKernelRegister<OpTypeCase,                            \
                                KernelType<DeviceType::kCPU, float>>   \
      g_##KernelType##_cpu_float_regst_var;                            \
  static CpuDoubleKernelRegister<OpTypeCase,                           \
                                 KernelType<DeviceType::kCPU, double>> \
      g_##KernelType##_cpu_double_regst_var;

#define REGISTER_GPU_KERNEL(OpTypeCase, KernelType)                    \
  static GpuFloatKernelRegister<OpTypeCase,                            \
                                KernelType<DeviceType::kGPU, float>>   \
      g_##KernelType##_gpu_float_regst_var;                            \
  static GpuDoubleKernelRegister<OpTypeCase,                           \
                                 KernelType<DeviceType::kGPU, double>> \
      g_##KernelType##_gpu_double_regst_var;

//#define REGISTER_KERNEL(OpTypeCase, KernelType) \
//  REGISTER_CPU_KERNEL(OpTypeCase, KernelType)   \
//  REGISTER_GPU_KERNEL(OpTypeCase, KernelType)
#define REGISTER_KERNEL(OpTypeCase, KernelType) \
  REGISTER_CPU_KERNEL(OpTypeCase, KernelType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_
