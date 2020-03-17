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
