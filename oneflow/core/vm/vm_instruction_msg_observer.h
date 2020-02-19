#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_

#include "oneflow/core/common/object_msg.h"

namespace oneflow {

class VpuId;

class VmInstructionMsgObserver {
 public:
  virtual ~VmInstructionMsgObserver() = default;

  virtual void OnDispatched(const VpuId &) = 0;
  virtual void OnReady(const VpuId &) = 0;
  virtual void OnDone(const VpuId &) = 0;

 protected:
  VmInstructionMsgObserver() = default;
};

class ObjectMsgAllocator;

class VmInstructionMsgNoneObserver : public VmInstructionMsgObserver {
 public:
  VmInstructionMsgNoneObserver() : VmInstructionMsgObserver() {}
  ~VmInstructionMsgNoneObserver() override = default;

  void OnDispatched(const VpuId &) override{};
  void OnReady(const VpuId &) override{};
  void OnDone(const VpuId &) override{};

  static VmInstructionMsgNoneObserver *Singleton() {
    static VmInstructionMsgNoneObserver observer;
    return &observer;
  }
  static VmInstructionMsgNoneObserver *NewObserver(ObjectMsgAllocator *, int32_t *size) {
    *size = 0;
    return Singleton();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_
