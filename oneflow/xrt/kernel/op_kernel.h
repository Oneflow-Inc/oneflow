/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_KERNEL_OP_KERNEL_H_
#define ONEFLOW_XRT_KERNEL_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_context.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

template<typename TContext>
class OpKernel {
 public:
  virtual void Compile(TContext *ctx) = 0;

  OpKernel() = default;
  virtual ~OpKernel() = default;
};

template<typename TContext>
class OpKernelRegistrar {
 private:
  std::function<OpKernel<TContext> *()> factory_;

  std::string op_name_;

  XrtEngine engine_field_;
  std::vector<XrtDevice> device_{CPU_X86, GPU_CUDA};

  // Attributes releated to the op kernel.
  bool train_phase_enabled_ = false;
  bool is_optimizer_op_ = false;
  util::Set<std::string> mutable_variables_ = {};

 public:
  explicit OpKernelRegistrar(const std::string &name) : op_name_(name) {}

  virtual ~OpKernelRegistrar() = default;

  auto FieldRegistry() -> util::FieldRegistry<XrtField, std::string, decltype(factory_)> * {
    return util::FieldRegistry<XrtField, std::string, decltype(factory_)>::Global();
  }

  OpKernelRegistrar &SetFactory(decltype(factory_) factory) {
    factory_ = factory;
    return *this;
  }

  OpKernelRegistrar &SetField(const XrtEngine &field) {
    engine_field_ = field;
    return *this;
  }

  OpKernelRegistrar &SetDevice(const std::vector<XrtDevice> &device) {
    device_ = device;
    return *this;
  }

  OpKernelRegistrar &SetMutableVariables(const util::Set<std::string> &variables) {
    mutable_variables_ = variables;
    return *this;
  }

  OpKernelRegistrar &SetIsOptimizerOp(const bool &is_optimizer) {
    is_optimizer_op_ = is_optimizer;
    return *this;
  }

  OpKernelRegistrar &EnableTrainPhase() {
    train_phase_enabled_ = true;
    return *this;
  }

  OpKernelRegistrar &Finalize() {
    util::Map<std::string, Any> attributes;
    attributes[TrainPhaseEnabledAttrName] = train_phase_enabled_;
    attributes[IsOptimizerOpAttrName] = is_optimizer_op_;
    attributes[MutableVariablesAttrName] = mutable_variables_;

    for (const auto &device : device_) {
      XrtField field = MakeXrtField(device, engine_field_);
      FieldRegistry()->Get(field)->Register(op_name_, factory_, attributes);
    }
    return *this;
  }
};

template<typename TContext>
struct OpKernelBuilder {
  OpKernel<TContext> *operator()(const XrtField &field, const std::string &op_name) {
    return util::FieldRegistry<XrtField, std::string,
                               std::function<OpKernel<TContext> *()>>::Global()
        ->Get(field)
        ->Lookup(op_name)();
  }
};

inline bool OpKernelRegistered(const std::string &op_type, const XrtField &field) {
  auto *rm = util::RegistryManager<XrtField>::Global();
  if (rm->HasRegistry(field)) { return rm->GetRegistry(field)->IsRegistered(op_type); }
  return false;
}

template<typename T>
inline const T &LookupOpKernelAttr(const std::string &op_type, const XrtField &field,
                                   const std::string &attr_name) {
  auto *rm = util::RegistryManager<XrtField>::Global();
  CHECK(rm->HasRegistry(field)) << "Field registry has not been found. Field is (engine: "
                                << field.engine() << ", device: " << field.device() << ").";
  const auto &attrs = rm->GetRegistry(field)->LookupAttr(op_type);
  CHECK_GT(attrs.count(attr_name), 0)
      << "Attribute " << attr_name << " is not found for OpKernel " << op_type;
  return any_cast<T>(attrs.at(attr_name));
}

inline const util::Set<std::string> &MutableVariables(const std::string &op_type,
                                                      const XrtField &field) {
  return LookupOpKernelAttr<util::Set<std::string>>(op_type, field, MutableVariablesAttrName);
}

inline const bool &TrainPhaseEnabled(const std::string &op_type, const XrtField &field) {
  return LookupOpKernelAttr<bool>(op_type, field, TrainPhaseEnabledAttrName);
}

inline const bool &IsOptimizerOp(const std::string &op_type, const XrtField &field) {
  return LookupOpKernelAttr<bool>(op_type, field, IsOptimizerOpAttrName);
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_KERNEL_OP_KERNEL_H_
