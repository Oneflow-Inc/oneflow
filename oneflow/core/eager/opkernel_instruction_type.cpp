#include "oneflow/core/common/util.h"
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
    const auto& job_desc_object =
        instruction->operand_type(view->job_desc()).Get<vm::ObjectWrapper<JobDesc>>();
    for (int i = 0; i < view->op_size(); ++i) {
      CHECK_GT(view->op(i).logical_object_id(), 0);
      const auto& op_conf_object =
          instruction->operand_type(view->op_conf(i)).Get<vm::ObjectWrapper<OperatorConf>>();
      CHECK(op_conf_object->has_user_conf());
      CHECK(op_conf_object->user_conf().input().empty());
      CHECK(op_conf_object->user_conf().output().empty());
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
COMMAND(
    vm::RegisterLocalInstructionType<InitOpKernelObjectInstructionType>("LocalInitOpKernelObject"));

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
COMMAND(vm::RegisterLocalInstructionType<DeleteOpKernelObjectInstructionType>(
    "DeleteLocalOpKernelObject"));

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
void ForEachIbnAndLogicalObjectId(const vm::Instruction& instruction, const T& args,
                                  const CallbackT& Callback) {
  CHECK_EQ(args.ibn_size(), args.input_index_size());
  CHECK_EQ(args.ibn_size(), args.input_blob_size());
  FOR_RANGE(int, i, 0, args.ibn_size()) {
    const std::string& bn_in_op =
        instruction.operand_type(args.ibn(i)).template Get<vm::StringObject>().str();
    int64_t index = args.input_index(i);
    int64_t logical_object_id = args.input_blob(i).logical_object_id();
    Callback(bn_in_op, index, logical_object_id);
  }
}

template<typename T, typename CallbackT>
void ForEachIbnAndBlobObject(vm::Instruction* instruction, const T& args,
                             const CallbackT& Callback) {
  CHECK_EQ(args.ibn_size(), args.input_index_size());
  CHECK_EQ(args.ibn_size(), args.input_blob_size());
  FOR_RANGE(int, i, 0, args.ibn_size()) {
    const std::string& bn_in_op =
        instruction->operand_type(args.ibn(i)).template Get<vm::StringObject>().str();
    int64_t index = args.input_index(i);
    const auto& blob_object =
        instruction->operand_type(args.input_blob(i)).template Get<BlobObject>();
    Callback(GenRepeatedBn(bn_in_op, index), blob_object);
  }
}

template<typename T, typename CallbackT>
void ForEachObnAndLogicalObjectId(const vm::Instruction& instruction, const T& args,
                                  const CallbackT& Callback) {
  CHECK_EQ(args.obn_size(), args.output_index_size());
  CHECK_EQ(args.obn_size(), args.output_blob_size());
  FOR_RANGE(int, i, 0, args.obn_size()) {
    const std::string& bn_in_op =
        instruction.operand_type(args.obn(i)).template Get<vm::StringObject>().str();
    int64_t index = args.output_index(i);
    int64_t logical_object_id = args.output_blob(i).logical_object_id();
    Callback(bn_in_op, index, logical_object_id);
  }
  CHECK_EQ(args.mut2_obn_size(), args.mut2_output_index_size());
  CHECK_EQ(args.mut2_obn_size(), args.mut2_output_blob_size());
  FOR_RANGE(int, i, 0, args.mut2_obn_size()) {
    const std::string& bn_in_op =
        instruction.operand_type(args.mut2_obn(i)).template Get<vm::StringObject>().str();
    int64_t index = args.mut2_output_index(i);
    int64_t logical_object_id = args.mut2_output_blob(i).logical_object_id();
    Callback(bn_in_op, index, logical_object_id);
  }
}

template<typename T, typename CallbackT>
void ForEachObnAndBlobObject(vm::Instruction* instruction, const T& args,
                             const CallbackT& Callback) {
  CHECK_EQ(args.obn_size(), args.output_index_size());
  CHECK_EQ(args.obn_size(), args.output_blob_size());
  FOR_RANGE(int, i, 0, args.obn_size()) {
    const std::string& bn_in_op =
        instruction->operand_type(args.obn(i)).template Get<vm::StringObject>().str();
    int64_t index = args.output_index(i);
    auto* blob_object =
        instruction->mut_operand_type(args.output_blob(i))->template Mut<BlobObject>();
    Callback(GenRepeatedBn(bn_in_op, index), blob_object);
  }
  CHECK_EQ(args.mut2_obn_size(), args.mut2_output_index_size());
  CHECK_EQ(args.mut2_obn_size(), args.mut2_output_blob_size());
  FOR_RANGE(int, i, 0, args.mut2_obn_size()) {
    const std::string& bn_in_op =
        instruction->operand_type(args.mut2_obn(i)).template Get<vm::StringObject>().str();
    int64_t index = args.mut2_output_index(i);
    auto* blob_object =
        instruction->mut_operand_type(args.mut2_output_blob(i))->template Mut<BlobObject>();
    Callback(GenRepeatedBn(bn_in_op, index), blob_object);
  }
}

template<typename T>
std::function<BlobDesc*(const std::string& bn_in_op)> MakeBlobDesc4BnInOp(
    vm::Instruction* instruction, const T& args, OpKernelObject* opkernel_obj) {
  const auto& obn2blob_desc = std::make_shared<HashMap<std::string, BlobDesc*>>();
  {
    HashSet<const BlobDesc*> out_blob_descs;
    ForEachObnAndBlobObject(instruction, args,
                            [&](const std::string& bn_in_op, BlobObject* blob_object) {
                              auto* blob_desc = blob_object->mut_blob_desc();
                              CHECK(out_blob_descs.insert(blob_desc).second);
                              CHECK(obn2blob_desc->emplace(bn_in_op, blob_desc).second);
                            });
  }
  const auto& ibn2blob_desc = std::make_shared<HashMap<std::string, const BlobDesc*>>();
  ForEachIbnAndBlobObject(
      instruction, args, [&](const std::string& bn_in_op, const BlobObject& blob_object) {
        CHECK(ibn2blob_desc->emplace(bn_in_op, &blob_object.blob_desc()).second);
      });
  const std::string tmp_bn = GenRepeatedBn("tmp_buffer", 0);
  BlobDesc* tmp = opkernel_obj->mut_tmp_buffer_blob_object()->mut_blob_desc();
  return [obn2blob_desc, ibn2blob_desc, tmp_bn, tmp](const std::string& bn_in_op) -> BlobDesc* {
    const auto& output_iter = obn2blob_desc->find(bn_in_op);
    if (output_iter != obn2blob_desc->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob_desc->find(bn_in_op);
    if (input_iter != ibn2blob_desc->end()) { return const_cast<BlobDesc*>(input_iter->second); }
    if (tmp_bn == bn_in_op) { return tmp; }
    return nullptr;
  };
}

template<typename T>
std::function<Blob*(const std::string& bn_in_op)> MakeBlob4BnInOp(vm::Instruction* instruction,
                                                                  const T& args,
                                                                  OpKernelObject* opkernel_obj) {
  const auto& obn2blob = std::make_shared<HashMap<std::string, Blob*>>();
  ForEachObnAndBlobObject(instruction, args,
                          [&](const std::string& bn_in_op, BlobObject* blob_object) {
                            CHECK(obn2blob->emplace(bn_in_op, blob_object->mut_blob()).second);
                          });
  const auto& ibn2blob = std::make_shared<HashMap<std::string, const Blob*>>();
  ForEachIbnAndBlobObject(instruction, args,
                          [&](const std::string& bn_in_op, const BlobObject& blob_object) {
                            CHECK(ibn2blob->emplace(bn_in_op, &blob_object.blob()).second);
                          });
  const std::string tmp_bn = GenRepeatedBn("tmp_buffer", 0);
  Blob* tmp = opkernel_obj->mut_tmp_buffer_blob_object()->mut_blob();
  return [obn2blob, ibn2blob, tmp_bn, tmp](const std::string& bn_in_op) -> Blob* {
    const auto& output_iter = obn2blob->find(bn_in_op);
    if (output_iter != obn2blob->end()) { return output_iter->second; }
    const auto& input_iter = ibn2blob->find(bn_in_op);
    if (input_iter != ibn2blob->end()) { return const_cast<Blob*>(input_iter->second); }
    if (tmp_bn == bn_in_op) { return tmp; }
    return nullptr;
  };
}

template<typename T>
void InitOutputBlobObjects(vm::Instruction* instruction, const T& args, int64_t device_id,
                           DataType data_type) {
  const auto& InitRwMutexedObject = [&](vm::RwMutexedObject* rw_mutexed_object) {
    const auto& parallel_desc = instruction->parallel_desc();
    CHECK(static_cast<bool>(parallel_desc));
    DeviceType device_type = parallel_desc->device_type();
    const auto& mem_case = MakeMemCase(device_type, device_id);
    rw_mutexed_object->Init<BlobObject>(mem_case, data_type);
  };
  FOR_RANGE(int, i, 0, args.output_blob_size()) {
    InitRwMutexedObject(instruction->mut_operand_type(args.output_blob(i)));
  }
  FOR_RANGE(int, i, 0, args.mut2_output_blob_size()) {
    InitRwMutexedObject(instruction->mut_operand_type(args.mut2_output_blob(i)));
  }
}

template<typename T>
void UpdateUserOpConfInputAndOutput(const vm::Instruction& instruction, UserOpConf* user_op_conf,
                                    const std::string& op_name, const T& args) {
  user_op_conf->clear_input();
  ForEachIbnAndLogicalObjectId(instruction, args,
                               [&](const std::string& ibn, int64_t i, int64_t logical_object_id) {
                                 auto* str_list = &(*user_op_conf->mutable_input())[ibn];
                                 CHECK_EQ(str_list->s_size(), i);
                                 str_list->add_s(op_name + "/" + GenRepeatedBn(ibn, i));
                               });
  user_op_conf->clear_output();
  ForEachObnAndLogicalObjectId(instruction, args,
                               [&](const std::string& obn, int64_t i, int64_t logical_object_id) {
                                 auto* str_list = &(*user_op_conf->mutable_output())[obn];
                                 CHECK_EQ(str_list->s_size(), i);
                                 str_list->add_s(op_name + "/" + GenRepeatedBn(obn, i));
                               });
}

void ResetTmpBufferBlobObject(OpKernelObject* opkernel_obj, DeviceType device_type,
                              int64_t device_id, DataType default_data_type) {
  const auto& mem_case = MakeMemCase(device_type, device_id);
  opkernel_obj->reset_tmp_buffer_blob_object(new BlobObject(mem_case, default_data_type));
}

template<typename T>
void OpKernelInfer(OpKernelObject* opkernel_obj, vm::Instruction* instruction, const T& args,
                   DeviceType device_type) {
  {
    DataType default_data_type = opkernel_obj->job_desc().DefaultDataType();
    int64_t device_id = instruction->stream().device_id();
    InitOutputBlobObjects(instruction, args, device_id, default_data_type);
    ResetTmpBufferBlobObject(opkernel_obj, device_type, device_id, default_data_type);
  }
  UpdateUserOpConfInputAndOutput(*instruction, opkernel_obj->mut_user_op_conf(),
                                 opkernel_obj->op_name(), args);
  opkernel_obj->ResetOpAndKernel(MakeBlobDesc4BnInOp(instruction, args, opkernel_obj));
  ForEachObnAndBlobObject(instruction, args, [](const std::string& _, BlobObject* blob_object) {
    blob_object->mutable_blob();
  });
  if (opkernel_obj->tmp_buffer_blob_object().blob_desc().shape().elem_cnt() > 0) {
    opkernel_obj->mut_tmp_buffer_blob_object()->mutable_blob();
  }
  opkernel_obj->kernel().Infer(MakeBlob4BnInOp(instruction, args, opkernel_obj));
}

template<typename T>
void OpKernelCompute(OpKernelObject* opkernel_obj, vm::Instruction* instruction, const T& args) {
  const auto& Blob4BnInOp = MakeBlob4BnInOp(instruction, args, opkernel_obj);
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  ForEachObnAndBlobObject(instruction, args, [&](const std::string&, BlobObject* blob_object) {
    blob_object->TryAllocateBlobBodyMemory(device_ctx);
  });
  if (opkernel_obj->mut_tmp_buffer_blob_object()->blob_desc().shape().elem_cnt() > 0) {
    opkernel_obj->mut_tmp_buffer_blob_object()->TryAllocateBlobBodyMemory(device_ctx);
  }
  std::shared_ptr<user_op::OpKernelState> new_state;
  {
    EagerKernel* eager_kernel = opkernel_obj->mut_kernel();
    const auto& old_state = opkernel_obj->opkernel_state();
    new_state = eager_kernel->EagerModelForward(old_state, device_ctx, Blob4BnInOp);
  }
  opkernel_obj->reset_opkernel_state(new_state);
}

}  // namespace

void CallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj = instruction->mut_operand_type(args->opkernel())->Mut<OpKernelObject>();
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(this->device_tag()));
  OpKernelInfer(opkernel_obj, instruction, args.Get(), device_type);
}

void CallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj = instruction->mut_operand_type(args->opkernel())->Mut<OpKernelObject>();
  OpKernelCompute(opkernel_obj, instruction, args.Get());
}

void StatelessCallOpKernelInstructionType::Infer(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  const auto& job_desc_ptr =
      instruction->operand_type(args->job_desc()).Get<vm::ObjectWrapper<JobDesc>>().GetPtr();
  const auto& op_conf =
      instruction->mut_operand_type(args->op_conf())->Get<vm::ObjectWrapper<OperatorConf>>().Get();
  CHECK(op_conf.has_user_conf());
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(this->device_tag()));
  vm::RwMutexedObject* rw_mutexed_object = instruction->mut_operand_type(args->shared_opkernel());
  if (rw_mutexed_object->has_object()) { CHECK(rw_mutexed_object->Has<OpKernelObject>()); }
  const auto& parallel_desc = instruction->parallel_desc();
  CHECK(static_cast<bool>(parallel_desc));
  CHECK_EQ(device_type, parallel_desc->device_type());
  rw_mutexed_object->reset_object();
  auto* opkernel_obj = rw_mutexed_object->Init<OpKernelObject>(op_conf, job_desc_ptr, device_type);
  OpKernelInfer(opkernel_obj, instruction, args.Get(), device_type);
}

void StatelessCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  FlatMsgView<StatelessCallOpKernelInstrOperand> args(instruction->instr_msg().operand());
  auto* opkernel_obj =
      instruction->mut_operand_type(args->shared_opkernel())->Mut<OpKernelObject>();
  OpKernelCompute(opkernel_obj, instruction, args.Get());
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
