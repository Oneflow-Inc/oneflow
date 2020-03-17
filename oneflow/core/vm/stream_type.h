#ifndef ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_

#include <string>
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instr_type_id.msg.h"
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

class StreamType {
 public:
  virtual ~StreamType() = default;

  virtual void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const = 0;

  virtual void InitInstructionStatus(const Stream& stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual void Run(InstrChain* instr_chain) const = 0;

  virtual ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                        int64_t this_machine_id) const = 0;
  virtual ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const {
    return ObjectMsgPtr<StreamDesc>();
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

const StreamType* LookupStreamType(StreamTypeId);
void RegisterStreamType(StreamTypeId, const StreamType*);
template<typename T>
void RegisterStreamType() {
  RegisterStreamType(T::kStreamTypeId, new T());
}

class InstrTypeId;

const InstrTypeId& LookupInstrTypeId(const std::string& instr_type_name);
void RegisterInstrTypeId(const std::string& instr_type_name, StreamTypeId stream_type_id,
                         InstructionOpcode opcode, VmType type);
template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, InstructionOpcode opcode,
                         VmType type) {
  RegisterInstrTypeId(instr_type_name, T::kStreamTypeId, opcode, type);
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_TYPE_H_
