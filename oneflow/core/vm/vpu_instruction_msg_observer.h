#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_

#include "oneflow/core/common/object_msg.h"

namespace oneflow {

class VpuId;

class VpuInstructionMsgObserver {
 public:
  virtual ~VpuInstructionMsgObserver() = default;

  virtual void OnDispatched(const VpuId &) = 0;
  virtual void OnReady(const VpuId &) = 0;
  virtual void OnDone(const VpuId &) = 0;

 protected:
  VpuInstructionMsgObserver();
};

class ObjectMsgAllocator;

class VpuInstructionMsgNoneObserver : public VpuInstructionMsgObserver {
 public:
  VpuInstructionMsgNoneObserver() : VpuInstructionMsgObserver() {}
  ~VpuInstructionMsgNoneObserver() override = default;

  void OnDispatched(const VpuId &) override{};
  void OnReady(const VpuId &) override{};
  void OnDone(const VpuId &) override{};

  static VpuInstructionMsgNoneObserver *Singleton() {
    static VpuInstructionMsgNoneObserver observer;
    return &observer;
  }
  static VpuInstructionMsgNoneObserver *NewObserver(ObjectMsgAllocator *, int32_t *size) {
    *size = 0;
    return Singleton();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_
