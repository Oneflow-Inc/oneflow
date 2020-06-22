#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/job/foreign_watcher.h"
#include "oneflow/core/register/ofblob.h"

namespace oneflow {
namespace eager {

class InitOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  InitOpKernelObjectInstructionType() = default;
  ~InitOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::Instruction* instruction) const override {
    FlatMsgView<NewOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    CHECK_EQ(view->op_conf_size(), view->op_size());
    const auto* operand_job_desc = instruction->operand_type(view->job_desc());
    CHECK_NOTNULL(operand_job_desc);
    const auto& job_desc_object = operand_job_desc->Get<vm::ObjectWrapper<JobDesc>>();
    for (int i = 0; i < view->op_size(); ++i) {
      CHECK_GT(view->op(i).logical_object_id(), 0);
      const auto* operand_op_conf = instruction->operand_type(view->op_conf(i));
      CHECK_NOTNULL(operand_op_conf);
      const auto& op_conf_object = operand_op_conf->Get<vm::ObjectWrapper<OperatorConf>>();
      CHECK(op_conf_object->has_user_conf());
      vm::RwMutexedObject* rw_mutexed_object = instruction->mut_operand_type(view->op(i));
      const auto& parallel_desc = instruction->parallel_desc();
      CHECK(static_cast<bool>(parallel_desc));
      DeviceType device_type = parallel_desc->device_type();
      rw_mutexed_object->Init<OpKernelObject>(op_conf_object.Get(), job_desc_object.GetPtr(),
                                              device_type);
    }
  }
  void Compute(vm::Instruction* instruction) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<InitOpKernelObjectInstructionType>("InitOpKernelObject"));

class DeleteOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteOpKernelObjectInstructionType() = default;
  ~DeleteOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::Instruction* instruction) const override {
    FlatMsgView<DeleteOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    for (int i = 0; i < view->op_size(); ++i) {
      auto* type_rw_mutexed_object = instruction->mut_operand_type(view->op(i));
      CHECK(type_rw_mutexed_object->Has<OpKernelObject>());
      type_rw_mutexed_object->reset_object();
    }
  }
  void Compute(vm::Instruction* instruction) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<DeleteOpKernelObjectInstructionType>("DeleteOpKernelObject"));

namespace {

std::shared_ptr<MemoryCase> MakeMemCase(const DeviceType device_type, const int64_t device_id) {
  const auto& mem_case = std::make_shared<MemoryCase>();
  if (device_type == DeviceType::kCPU) {
    mem_case->mutable_host_mem();
  } else if (device_type == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(device_id);
  } else {
    UNIMPLEMENTED();
  }
  return mem_case;
}

template<typename T, typename CallbackT>
Maybe<void> ForEachIbnAndBlobObject(vm::Instruction* instruction, const T& args,
                                    const CallbackT& Callback) {
  CHECK_EQ_OR_RETURN(args.ibn_size(), args.input_blob_size());
  FOR_RANGE(int, i, 0, args.ibn_size()) {
    const auto* operand_ibn = instruction->operand_type(args.ibn(i));
    CHECK_NOTNULL_OR_RETURN(operand_ibn);
    const std::string& bn_in_op = operand_ibn->template Get<vm::StringObject>().str();
    const auto* operand_input_blob = instruction->operand_type(args.input_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_input_blob)
        << "bn_in_op: " << bn_in_op << ", object_id: " << args.input_blob(i).logical_object_id();
    const auto& blob_object = operand_input_blob->template Get<BlobObject>();
    JUST(Callback(bn_in_op, blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename T, typename CallbackT>
Maybe<void> ForEachObnAndBlobObject(vm::Instruction* instruction, const T& args,
                                    const CallbackT& Callback) {
  CHECK_EQ_OR_RETURN(args.obn_size(), args.output_blob_size());
  FOR_RANGE(int, i, 0, args.obn_size()) {
    const auto* operand_obn = instruction->operand_type(args.obn(i));
    CHECK_NOTNULL_OR_RETURN(operand_obn);
    const std::string& bn_in_op = operand_obn->template Get<vm::StringObject>().str();
    auto* operand_output_blob = instruction->mut_operand_type(args.output_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_output_blob) << "obn: " << bn_in_op;
    auto* blob_object = operand_output_blob->template Mut<BlobObject>();
    JUST(Callback(bn_in_op, blob_object));
  }
  CHECK_EQ_OR_RETURN(args.mut2_obn_size(), args.mut2_output_blob_size());
  FOR_RANGE(int, i, 0, args.mut2_obn_size()) {
    const auto* operand_obn = instruction->operand_type(args.mut2_obn(i));
    CHECK_NOTNULL_OR_RETURN(operand_obn);
    const std::string& bn_in_op = operand_obn->template Get<vm::StringObject>().str();
    auto* operand_output_blob = instruction->mut_operand_type(args.mut2_output_blob(i));
    CHECK_NOTNULL_OR_RETURN(operand_output_blob) << "obn: " << bn_in_op;
    auto* blob_object = operand_output_blob->template Mut<BlobObject>();
    JUST(Callback(bn_in_op, blob_object));
  }
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MakeBlobDesc4BnInOp(vm::Instruction* instruction, const T& args,
                                OpKernelObject* opkernel_obj,
                                std::function<BlobDesc*(const std::string&)>* BlobDesc4BnInOp) {
  const auto& obn2blob_desc = std::make_shared<HashMap<std::string, BlobDesc*>>();
  {
    HashSet<const BlobDesc*> out_blob_descs;
    JUST(ForEachObnAndBlobObject(
        instruction, args,
        [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
          auto* blob_desc = blob_object->mut_blob_desc();
          CHECK_OR_RETURN(out_blob_descs.insert(blob_desc).second);
          CHECK_OR_RETURN(obn2blob_desc->emplace(bn_in_op, blob_desc).second);
          return Maybe<void>::Ok();
        }));
  }
  const auto& ibn2blob_desc = std::make_shared<HashMap<std::string, const BlobDesc*>>();
  JUST(ForEachIbnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob_desc->emplace(bn_in_op, &blob_object.blob_desc()).second);
        return Maybe<void>::Ok();
      }));
  const std::string tmp_bn = GenRepeatedBn("tmp_buffer", 0);
  BlobDesc* tmp = opkernel_obj->mut_tmp_buffer_blob_object()->mut_blob_desc();
  *BlobDesc4BnInOp = [obn2blob_desc, ibn2blob_desc, tmp_bn,
                      tmp](const std::string& bn_in_op) -> BlobDesc* {
    const auto& output_iter = obn2blob_desc->find(bn_in_op);
    if (output_iter != obn2blob_desc->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob_desc->find(bn_in_op);
    if (input_iter != ibn2blob_desc->end()) { return const_cast<BlobDesc*>(input_iter->second); }
    if (tmp_bn == bn_in_op) { return tmp; }
    return nullptr;
  };
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MakeBlobDesc4BnInOp(vm::Instruction* instruction, const T& args,
                                SystemOpKernelObject* opkernel_obj,
                                std::function<BlobDesc*(const std::string&)>* BlobDesc4BnInOp) {
  const auto& obn2blob_desc = std::make_shared<HashMap<std::string, BlobDesc*>>();
  {
    HashSet<const BlobDesc*> out_blob_descs;
    JUST(ForEachObnAndBlobObject(
        instruction, args,
        [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
          auto* blob_desc = blob_object->mut_blob_desc();
          CHECK_OR_RETURN(out_blob_descs.insert(blob_desc).second);
          CHECK_OR_RETURN(obn2blob_desc->emplace(bn_in_op, blob_desc).second);
          return Maybe<void>::Ok();
        }));
  }
  const auto& ibn2blob_desc = std::make_shared<HashMap<std::string, const BlobDesc*>>();
  JUST(ForEachIbnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob_desc->emplace(bn_in_op, &blob_object.blob_desc()).second);
        return Maybe<void>::Ok();
      }));
  *BlobDesc4BnInOp = [obn2blob_desc, ibn2blob_desc](const std::string& bn_in_op) -> BlobDesc* {
    const auto& output_iter = obn2blob_desc->find(bn_in_op);
    if (output_iter != obn2blob_desc->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob_desc->find(bn_in_op);
    if (input_iter != ibn2blob_desc->end()) { return const_cast<BlobDesc*>(input_iter->second); }
    return nullptr;
  };
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MakeBlob4BnInOp(vm::Instruction* instruction, const T& args,
                            OpKernelObject* opkernel_obj,
                            std::function<Blob*(const std::string&)>* Blob4BnInOp) {
  const auto& obn2blob = std::make_shared<HashMap<std::string, Blob*>>();
  JUST(ForEachObnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(obn2blob->emplace(bn_in_op, blob_object->mut_blob()).second);
        return Maybe<void>::Ok();
      }));
  const auto& ibn2blob = std::make_shared<HashMap<std::string, const Blob*>>();
  JUST(ForEachIbnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob->emplace(bn_in_op, &blob_object.blob()).second);
        return Maybe<void>::Ok();
      }));
  const std::string tmp_bn = GenRepeatedBn("tmp_buffer", 0);
  Blob* tmp = opkernel_obj->mut_tmp_buffer_blob_object()->mut_blob();
  *Blob4BnInOp = [obn2blob, ibn2blob, tmp_bn, tmp](const std::string& bn_in_op) -> Blob* {
    const auto& output_iter = obn2blob->find(bn_in_op);
    if (output_iter != obn2blob->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob->find(bn_in_op);
    if (input_iter != ibn2blob->end()) { return const_cast<Blob*>(input_iter->second); }
    if (tmp_bn == bn_in_op) { return tmp; }
    return nullptr;
  };
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MakeBlob4BnInOp(vm::Instruction* instruction, const T& args,
                            SystemOpKernelObject* opkernel_obj,
                            std::function<Blob*(const std::string&)>* Blob4BnInOp) {
  const auto& obn2blob = std::make_shared<HashMap<std::string, Blob*>>();
  JUST(ForEachObnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, BlobObject* blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(obn2blob->emplace(bn_in_op, blob_object->mut_blob()).second);
        return Maybe<void>::Ok();
      }));
  const auto& ibn2blob = std::make_shared<HashMap<std::string, const Blob*>>();
  JUST(ForEachIbnAndBlobObject(
      instruction, args,
      [&](const std::string& bn_in_op, const BlobObject& blob_object) -> Maybe<void> {
        CHECK_OR_RETURN(ibn2blob->emplace(bn_in_op, &blob_object.blob()).second);
        return Maybe<void>::Ok();
      }));
  *Blob4BnInOp = [obn2blob, ibn2blob](const std::string& bn_in_op) -> Blob* {
    const auto& output_iter = obn2blob->find(bn_in_op);
    if (output_iter != obn2blob->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob->find(bn_in_op);
    if (input_iter != ibn2blob->end()) { return const_cast<Blob*>(input_iter->second); }
    return nullptr;
  };
  return Maybe<void>::Ok();
}

template<typename T>
void InitOutputBlobObjects(vm::Instruction* instruction, const T& args,
                           const std::shared_ptr<MemoryCase>& mem_case, DataType data_type) {
  const auto& InitRwMutexedObject = [&](vm::RwMutexedObject* rw_mutexed_object) {
    const auto& parallel_desc = instruction->parallel_desc();
    CHECK(static_cast<bool>(parallel_desc));
    if (rw_mutexed_object->has_object()) {
      CHECK(rw_mutexed_object->Has<BlobObject>());
    } else {
      rw_mutexed_object->Init<BlobObject>(mem_case, data_type);
    }
  };
  FOR_RANGE(int, i, 0, args.output_blob_size()) {
    InitRwMutexedObject(instruction->mut_operand_type(args.output_blob(i)));
  }
  FOR_RANGE(int, i, 0, args.mut2_output_blob_size()) {
    InitRwMutexedObject(instruction->mut_operand_type(args.mut2_output_blob(i)));
  }
}

void ResetTmpBufferBlobObject(OpKernelObject* opkernel_obj, DeviceType device_type,
                              int64_t device_id, DataType default_data_type) {
  const auto& mem_case = MakeMemCase(device_type, device_id);
  opkernel_obj->reset_tmp_buffer_blob_object(new BlobObject(mem_case, default_data_type));
}

template<typename T>
Maybe<void> OpKernelInfer(OpKernelObject* opkernel_obj, vm::Instruction* instruction, const T& args,
                          const std::shared_ptr<MemoryCase>& mem_case, DeviceType device_type) {
  {
    DataType default_data_type = opkernel_obj->job_desc().DefaultDataType();
    InitOutputBlobObjects(instruction, args, mem_case, default_data_type);
    int64_t device_id = instruction->stream().device_id();
    ResetTmpBufferBlobObject(opkernel_obj, device_type, device_id, default_data_type);
  }
  std::function<BlobDesc*(const std::string&)> BlobDesc4BnInOp;
  JUST(MakeBlobDesc4BnInOp(instruction, args, opkernel_obj, &BlobDesc4BnInOp));
  opkernel_obj->ResetOpAndKernel(BlobDesc4BnInOp);
  JUST(ForEachObnAndBlobObject(instruction, args,
                               [](const std::string&, BlobObject* blob_object) -> Maybe<void> {
                                 blob_object->mutable_blob();
                                 return Maybe<void>::Ok();
                               }));
  if (opkernel_obj->tmp_buffer_blob_object().blob_desc().shape().elem_cnt() > 0) {
    opkernel_obj->mut_tmp_buffer_blob_object()->mutable_blob();
  }
  std::function<Blob*(const std::string&)> Blob4BnInOp;
  JUST(MakeBlob4BnInOp(instruction, args, opkernel_obj, &Blob4BnInOp));
  opkernel_obj->kernel().Infer(Blob4BnInOp);
  return Maybe<void>::Ok();
}

Maybe<void> OpKernelInfer(SystemOpKernelObject* opkernel_obj, vm::Instruction* instruction,
                          const StatelessCallOpKernelInstrOperand& args,
                          const std::shared_ptr<MemoryCase>& mem_case, DeviceType device_type) {
  {
    DataType default_data_type = opkernel_obj->job_desc().DefaultDataType();
    InitOutputBlobObjects(instruction, args, mem_case, default_data_type);
  }
  std::function<BlobDesc*(const std::string&)> BlobDesc4BnInOp;
  JUST(MakeBlobDesc4BnInOp(instruction, args, opkernel_obj, &BlobDesc4BnInOp));
  opkernel_obj->ResetKernel(BlobDesc4BnInOp);
  JUST(ForEachObnAndBlobObject(instruction, args,
                               [](const std::string&, BlobObject* blob_object) -> Maybe<void> {
                                 blob_object->mutable_blob();
                                 return Maybe<void>::Ok();
                               }));
  std::function<Blob*(const std::string&)> Blob4BnInOp;
  JUST(MakeBlob4BnInOp(instruction, args, opkernel_obj, &Blob4BnInOp));
  opkernel_obj->kernel().SystemForwardHeader(KernelCtx(), Blob4BnInOp);
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> OpKernelCompute(OpKernelObject* opkernel_obj, vm::Instruction* instruction,
                            const T& args) {
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  JUST(ForEachObnAndBlobObject(instruction, args,
                               [&](const std::string&, BlobObject* blob_object) -> Maybe<void> {
                                 blob_object->TryAllocateBlobBodyMemory(device_ctx);
                                 return Maybe<void>::Ok();
                               }));
  if (opkernel_obj->mut_tmp_buffer_blob_object()->blob_desc().shape().elem_cnt() > 0) {
    opkernel_obj->mut_tmp_buffer_blob_object()->TryAllocateBlobBodyMemory(device_ctx);
  }
  std::shared_ptr<user_op::OpKernelState> new_state;
  {
    std::function<Blob*(const std::string&)> Blob4BnInOp;
    JUST(MakeBlob4BnInOp(instruction, args, opkernel_obj, &Blob4BnInOp));
    EagerKernel* eager_kernel = opkernel_obj->mut_kernel();
    const auto& old_state = opkernel_obj->opkernel_state();
    new_state = eager_kernel->EagerModelForward(old_state, device_ctx, Blob4BnInOp);
  }
  opkernel_obj->reset_opkernel_state(new_state);
  return Maybe<void>::Ok();
}

Maybe<void> OpKernelCompute(SystemOpKernelObject* opkernel_obj, vm::Instruction* instruction,
                            const StatelessCallOpKernelInstrOperand& args) {
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  JUST(ForEachObnAndBlobObject(instruction, args,
                               [&](const std::string&, BlobObject* blob_object) -> Maybe<void> {
                                 blob_object->TryAllocateBlobBodyMemory(device_ctx);
                                 return Maybe<void>::Ok();
                               }));
  KernelCtx kernel_ctx;
  kernel_ctx.device_ctx = device_ctx;
  std::function<Blob*(const std::string&)> Blob4BnInOp;
  JUST(MakeBlob4BnInOp(instruction, args, opkernel_obj, &Blob4BnInOp));
  opkernel_obj->kernel().SystemForwardDataContent(kernel_ctx, Blob4BnInOp);
  return Maybe<void>::Ok();
}

template<typename T>
T* GetSharedOpKernel(vm::Instruction* instruction, DeviceType device_type,
                     const StatelessCallOpKernelInstrOperand& args) {
  const auto* operand_job_desc = instruction->operand_type(args.job_desc());
  CHECK_NOTNULL(operand_job_desc);
  const auto& job_desc_ptr = operand_job_desc->Get<vm::ObjectWrapper<JobDesc>>().GetPtr();
  const auto& op_conf =
      instruction->mut_operand_type(args.op_conf())->Get<vm::ObjectWrapper<OperatorConf>>().Get();
  vm::RwMutexedObject* rw_mutexed_object = instruction->mut_operand_type(args.shared_opkernel());
  CHECK(!rw_mutexed_object->has_object() || rw_mutexed_object->Has<OpKernelObject>()
        || rw_mutexed_object->Has<SystemOpKernelObject>());
  const auto& parallel_desc = instruction->parallel_desc();
  CHECK(static_cast<bool>(parallel_desc));
  CHECK_EQ(device_type, parallel_desc->device_type());
  rw_mutexed_object->reset_object();
  return rw_mutexed_object->Init<T>(op_conf, job_desc_ptr, device_type);
}

}  // namespace

void CallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj = instruction->mut_operand_type(args->opkernel())->Mut<OpKernelObject>();
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(this->device_tag()));
  int64_t device_id = instruction->stream().device_id();
  const auto& mem_case = MakeMemCase(device_type, device_id);
  CHECK_JUST(OpKernelInfer(opkernel_obj, instruction, args.Get(), mem_case, device_type));
}

void CallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj = instruction->mut_operand_type(args->opkernel())->Mut<OpKernelObject>();
  CHECK_JUST(OpKernelCompute(opkernel_obj, instruction, args.Get()));
}

void UserStatelessCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(this->device_tag()));
  int64_t device_id = instruction->stream().device_id();
  auto* opkernel = GetSharedOpKernel<OpKernelObject>(instruction, device_type, args.Get());
  const auto& mem_case = MakeMemCase(device_type, device_id);
  CHECK_JUST(OpKernelInfer(opkernel, instruction, args.Get(), mem_case, device_type));
}

void UserStatelessCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj =
      instruction->mut_operand_type(args->shared_opkernel())->Mut<OpKernelObject>();
  CHECK_JUST(OpKernelCompute(opkernel_obj, instruction, args.Get()));
}

std::shared_ptr<MemoryCase> SystemStatelessCallOpKernelInstructionType::GetOutBlobMemCase(
    const DeviceType device_type, const int64_t device_id) const {
  return MakeMemCase(device_type, device_id);
}

void SystemStatelessCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(this->device_tag()));
  int64_t device_id = instruction->stream().device_id();
  auto* opkernel = GetSharedOpKernel<SystemOpKernelObject>(instruction, device_type, args.Get());
  const auto& mem_case = GetOutBlobMemCase(device_type, device_id);
  CHECK_JUST(OpKernelInfer(opkernel, instruction, args.Get(), mem_case, device_type));
}

void SystemStatelessCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj =
      instruction->mut_operand_type(args->shared_opkernel())->Mut<SystemOpKernelObject>();
  CHECK_JUST(OpKernelCompute(opkernel_obj, instruction, args.Get()));
}

void WatchBlob(vm::Instruction* instruction) {
  FlatMsgView<WatchBlobInstrOperand> args(instruction->instr_msg().operand());
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  auto* blob_object = instruction->mut_operand_type(args->blob())->Mut<BlobObject>();
  OfBlob of_blob(device_ctx, blob_object->mut_blob());
  int64_t of_blob_ptr = reinterpret_cast<int64_t>(&of_blob);
  Global<ForeignWorkerCallback>::Get()->Call(args->unique_callback_id(), of_blob_ptr);
}

void WatchBlobHeaderInstructionType::Infer(vm::Instruction* instruction) const {
  WatchBlob(instruction);
}

void WatchBlobBodyInstructionType::Compute(vm::Instruction* instruction) const {
  WatchBlob(instruction);
}

}  // namespace eager
}  // namespace oneflow
