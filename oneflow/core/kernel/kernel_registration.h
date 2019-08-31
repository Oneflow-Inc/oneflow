#ifndef ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_reg_value.pb.h"

namespace oneflow {

class Kernel;

namespace kernel_registration {

using CreateFn = std::function<Kernel*()>;

namespace constraint {

class KernelConstraint {
 public:
  KernelConstraint() = default;
  virtual ~KernelConstraint() = default;

  virtual bool IsMatched(const KernelConf&) const = 0;
  virtual size_t PriorityLevel() const { return 0; }  // big number means high priority
  virtual void ToProto(KernelRegValProto::RegVal*) const = 0;
};

class NoConstraint final : public KernelConstraint {
 public:
  bool IsMatched(const KernelConf&) const override { return true; }
  void ToProto(KernelRegValProto::RegVal*) const override;
};

class DeviceAndDTypeConstraint final : public KernelConstraint {
 public:
  DeviceAndDTypeConstraint(DeviceType dev, DataType dtype) : dev_(dev), dtype_(dtype) {}
  bool IsMatched(const KernelConf&) const override;
  void ToProto(KernelRegValProto::RegVal*) const override;

 private:
  DeviceType dev_;
  DataType dtype_;
};

class DeviceConstraint final : public KernelConstraint {
 public:
  DeviceConstraint(DeviceType dev) : dev_(dev) {}
  bool IsMatched(const KernelConf&) const override;
  void ToProto(KernelRegValProto::RegVal*) const override;

 private:
  DeviceType dev_;
};

class PredAndLabelKernelConstraint final : public KernelConstraint {
 public:
  PredAndLabelKernelConstraint(DeviceType dev, DataType pred_type, DataType label_type)
      : dev_(dev), pred_type_(pred_type), label_type_(label_type) {}
  bool IsMatched(const KernelConf&) const override;
  void ToProto(KernelRegValProto::RegVal*) const override;

 private:
  DeviceType dev_;
  DataType pred_type_;
  DataType label_type_;
};

}  // namespace constraint

struct KernelRegistrar final {
  KernelRegistrar(const OperatorConf::OpTypeCase& op_type, constraint::KernelConstraint* cons,
                  CreateFn f);
};

Kernel* CreateKernel(const KernelConf& kernel_conf);

void ExportProtoFromKernelRegistry(KernelRegValProto*);

}  // namespace kernel_registration

#define REGISTER_KERNEL_WITH_NOTHING(op_type, ...)                                 \
  namespace {                                                                      \
  static kernel_registration::KernelRegistrar OF_PP_CAT(g_registrar, __COUNTER__)( \
      op_type, new kernel_registration::constraint::NoConstraint(),                \
      []() { return new __VA_ARGS__(); });                                         \
  }  // namespace

#define REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, device, dtype, ...)                      \
  namespace {                                                                                   \
  static kernel_registration::KernelRegistrar OF_PP_CAT(g_registrar, __COUNTER__)(              \
      op_type,                                                                                  \
      new kernel_registration::constraint::DeviceAndDTypeConstraint(device,                     \
                                                                    GetDataType<dtype>::value), \
      []() { return new __VA_ARGS__(); });                                                      \
  }  // namespace

#define REGISTER_KERNEL_WITH_DEVICE(op_type, device, ...)                          \
  namespace {                                                                      \
  static kernel_registration::KernelRegistrar OF_PP_CAT(g_registrar, __COUNTER__)( \
      op_type, new kernel_registration::constraint::DeviceConstraint(device),      \
      []() { return new __VA_ARGS__(); });                                         \
  }  // namespace

#define REGISTER_KERNEL_WITH_PRED_AND_LABEL(op_type, device, pred_type, label_type, ...) \
  namespace {                                                                            \
  static kernel_registration::KernelRegistrar OF_PP_CAT(g_registrar, __COUNTER__)(       \
      op_type,                                                                           \
      new kernel_registration::constraint::PredAndLabelKernelConstraint(                 \
          device, GetDataType<pred_type>::value, GetDataType<label_type>::value),        \
      []() { return new __VA_ARGS__(); });                                               \
  }  // namespace

#define REGISTER_KERNEL_HELPER_GPU_FLOATING(op_type, kernel)               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, float,  \
                                        kernel<DeviceType::kGPU, float>)   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, double, \
                                        kernel<DeviceType::kGPU, double>)
#define REGISTER_KERNEL_HELPER_GPU_HALF(op_type, kernel)                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, float16, \
                                        kernel<DeviceType::kGPU, float16>)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_
