#ifndef ONEFLOW_KERNEL_KERNEL_MANAGER_H_
#define ONEFLOW_KERNEL_KERNEL_MANAGER_H_

#include <utility>
#include <memory>
#include <string>
#include "common/util.h"
#include "common/proto_io.h"
#include "kernel/kernel.h"
#include "job/ofelf.pb.h"

namespace oneflow {

template<typename Dtype>
class KernelMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelMgr);
  ~KernelMgr() = default;

  static KernelMgr& Singleton() {
    static KernelMgr obj;
    return obj;
  }

  const Kernel<Dtype>* GetKernelFromOpName(const std::string& op_name) {
    return op_name2kernel_ptr_.at(op_name).get();
  }

  void InitFromOpProtos(const PbRpf<OperatorProto>& op_protos) {
    for (const OperatorProto& op_proto : op_protos) {
      const std::string& op_name = op_proto.op_conf().name();
      std::unique_ptr<Kernel<Dtype>> kernel_ptr(
          CreateKernel(op_proto.op_conf().op_type_case()));
      kernel_ptr->InitFromOpProto(op_proto);
      CHECK(op_name2kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
    }
  }

 private:
  KernelMgr() = default;
  HashMap<const std::string, std::unique_ptr<Kernel<Dtype>>> op_name2kernel_ptr_;
};

template<typename Dtype>
Kernel<Dtype>* CreateKernel(OperatorConf::OpTypeCase op_type_case);
void AddCpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
  std::function<Kernel<float>*()> creator);
void AddGpuFloatKernelCreator(OperatorConf::OpTypeCase op_type_case,
  std::function<Kernel<float>*()> creator);
void AddCpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
  std::function<Kernel<double>*()> creator);
void AddGpuDoubleKernelCreator(OperatorConf::OpTypeCase op_type_case,
  std::function<Kernel<double>*()> creator);

template<OperatorConf::OpTypeCase op_type_case, typename KernelType>
struct CpuFloatKernelRegister {
  CpuFloatKernelRegister() {
    AddCpuFloatKernelCreator(op_type_case, []() { return new KernelType; });
  }
};

template<OperatorConf::OpTypeCase op_type_case, typename KernelType>
struct GpuFloatKernelRegister {
  GpuFloatKernelRegister() {
    AddGpuFloatKernelCreator(op_type_case, []() { return new KernelType });
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
    AddGpuDoubleKernelCreator(op_type_case, []() { return new KernelType });
  }
};

#define REGISTER_CPU_KERNEL(OpTypeCase, KernelType) \
  static CpuFloatKernelRegister<OpTypeCase, KernelType<DeviceType::kCPU, float>> g_##KernelType##_float_regst_var; \
  static CpuDoubleKernelRegister<OpTypeCase, KernelType<DeviceType::kCPU, double>> g_##KernelType##_double_regst_var;
#define REGISTER_GPU_FLOAT_KERNEL(OpTypeCase, KernelType) \
  static GpuFloatKernelRegister<OpTypeCase, KernelType<DeviceType::kGPU, float>> g_##KernelType##_float_regst_var; \
  static GpuDoubleKernelRegister<OpTypeCase, KernelType<DeviceType::kGPU, double>> g_##KernelType##_double_regst_var;

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_MANAGER_H_
