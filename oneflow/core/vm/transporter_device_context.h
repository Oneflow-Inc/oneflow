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
#ifndef ONEFLOW_CORE_VM_TRANSPORTER_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_VM_TRANSPORTER_DEVICE_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/vm/transporter.h"

namespace oneflow {
namespace vm {

class TransporterDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransporterDeviceCtx);
  TransporterDeviceCtx(Transporter* transporter) : transporter_(transporter) {}
  ~TransporterDeviceCtx() override = default;

  const Transporter& transporter() const { return *transporter_; }
  Transporter* mut_transporter() { return transporter_.get(); }

 private:
  std::unique_ptr<vm::Transporter> transporter_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TRANSPORTER_DEVICE_CONTEXT_H_
