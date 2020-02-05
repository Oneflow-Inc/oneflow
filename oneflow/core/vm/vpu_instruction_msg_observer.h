#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_OBSERVER_H_

#include "oneflow/core/common/object_msg.h"

namespace oneflow {

class OBJECT_MSG_TYPE(VpuId);

class VpuInstructionMsgObserver {
 public:
  virtual ~VpuInstructionMsgObserver() = default;

  virtual void OnDispatched(const OBJECT_MSG_TYPE(VpuId) &) = 0;
  virtual void OnReady(const OBJECT_MSG_TYPE(VpuId) &) = 0;
  virtual void OnDone(const OBJECT_MSG_TYPE(VpuId) &) = 0;

 protected:
  VpuInstructionMsgObserver();
};

class ObjectMsgAllocator;

class VpuInstructionMsgNoneObserver : public VpuInstructionMsgObserver {
 public:
  VpuInstructionMsgNoneObserver() : VpuInstructionMsgObserver() {}
  ~VpuInstructionMsgNoneObserver() override = default;

  void OnDispatched(const OBJECT_MSG_TYPE(VpuId) &) override{};
  void OnReady(const OBJECT_MSG_TYPE(VpuId) &) override{};
  void OnDone(const OBJECT_MSG_TYPE(VpuId) &) override{};

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
