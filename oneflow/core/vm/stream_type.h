#ifndef ONEFLOW_CORE_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_STREAM_TYPE_H_

#include <string>
#include <typeindex>
#include <glog/logging.h>
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/vm_type.h"
#include "oneflow/core/common/callback.msg.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class ObjectMsgAllocator;

namespace vm {

class Stream;
class InstructionStatusBuffer;
class InstrChain;
class VirtualMachine;
class InstructionMsg;
class InstructionType;

class StreamType {
 public:
  virtual ~StreamType() = default;

  void Run(InstrChain* instr_chain) const;
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const;
  void Run(VirtualMachine* vm, InstrChain* instr_chain) const;

  virtual const char* device_tag() const = 0;

  virtual void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const = 0;

  virtual void InitInstructionStatus(const Stream& stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual void Compute(InstrChain* instr_chain) const = 0;
  virtual void Infer(InstrChain* instr_chain) const { LOG(FATAL) << "UNIMPLEMENTED"; }

  virtual ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                        int64_t this_machine_id) const = 0;
  virtual ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const {
    return ObjectMsgPtr<StreamDesc>();
  }

  virtual bool SharingVirtualMachineThread() const { return false; }
  virtual void Infer(VirtualMachine* vm, InstrChain* instr_chain) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Compute(VirtualMachine* vm, InstrChain* instr_chain) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }

  template<VmType vm_type, typename Enabled = void>
  struct MakeStreamDescUtil {};

  template<typename Enabled>
  struct MakeStreamDescUtil<VmType::kRemote, Enabled> {
    static ObjectMsgPtr<StreamDesc> Call(const StreamType& self, const Resource& resource,
                                         int64_t this_machine_id) {
      return self.MakeRemoteStreamDesc(resource, this_machine_id);
    }
  };
  template<typename Enabled>
  struct MakeStreamDescUtil<VmType::kLocal, Enabled> {
    static ObjectMsgPtr<StreamDesc> Call(const StreamType& self, const Resource& resource,
                                         int64_t this_machine_id) {
      return self.MakeLocalStreamDesc(resource);
    }
  };

  template<VmType vm_type>
  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource, int64_t this_machine_id) const {
    return MakeStreamDescUtil<vm_type>::Call(*this, resource, this_machine_id);
  }

 protected:
  StreamType() = default;
};

HashMap<std::type_index, const StreamType*>* StreamType4TypeIndex();

template<typename T>
const StreamType* LookupStreamType4TypeIndex() {
  return StreamType4TypeIndex()->at(typeid(T));
}

template<typename T>
void TryRegisterStreamType4TypeIndex() {
  auto* map = StreamType4TypeIndex();
  std::type_index type_index(typeid(T));
  if (map->find(type_index) == map->end()) { map->emplace(type_index, new T()); }
}

const StreamTypeId& LookupInferStreamTypeId(const StreamTypeId& compute_stream_type_id);
void TryRegisterInferStreamTypeId(const StreamType* infer_stream_type,
                                  const StreamType* compute_stream_type);
template<typename InferStreamType, typename ComputeStreamType>
void TryRegisterInferStreamTypeId() {
  TryRegisterInferStreamTypeId(LookupStreamType4TypeIndex<InferStreamType>(),
                               LookupStreamType4TypeIndex<ComputeStreamType>());
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_TYPE_H_
