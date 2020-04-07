#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/job_object.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class NewOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  NewOpKernelObjectInstructionType() = default;
  ~NewOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<NewOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instr_ctx->instr_msg().operand()));
    const auto& job_object = instr_ctx->operand_type(view->job()).Get<JobObject>();
    for (int i = 0; i < view->op_size(); ++i) {
      CHECK_GT(view->op(i).logical_object_id(), 0);
      const OperatorConf& op_conf =
          job_object.OpConf4LogicalObjectId(view->op(i).logical_object_id());
      instr_ctx->mut_operand_type(view->op(i))
          ->Mutable<OpKernelObject>(job_object.job_desc(), op_conf);
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<NewOpKernelObjectInstructionType>("NewOpKernelObject"));
COMMAND(
    vm::RegisterLocalInstructionType<NewOpKernelObjectInstructionType>("NewLocalOpKernelObject"));

class DeleteOpKernelObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteOpKernelObjectInstructionType() = default;
  ~DeleteOpKernelObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<DeleteOpKernelObjectInstrOperand> view;
    CHECK(view.Match(instr_ctx->instr_msg().operand()));
    for (int i = 0; i < view->op_size(); ++i) {
      auto* type_mirrored_object = instr_ctx->mut_operand_type(view->op(i));
      CHECK(type_mirrored_object->Has<OpKernelObject>());
      type_mirrored_object->reset_object();
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(vm::RegisterInstructionType<DeleteOpKernelObjectInstructionType>("DeleteOpKernelObject"));
COMMAND(vm::RegisterLocalInstructionType<DeleteOpKernelObjectInstructionType>(
    "DeleteLocalOpKernelObject"));

namespace {

template<typename CallbackT>
void ForEachIbnAndBlobObject(vm::InstrCtx* instr_ctx,
                             const FlatMsgView<CallOpKernelInstrOperand>& args,
                             const CallbackT& Callback) {
  CHECK_EQ(args->ibn_size(), args->input_index_size());
  CHECK_EQ(args->ibn_size(), args->input_blob_size());
  FOR_RANGE(int, i, 0, args->ibn_size()) {
    const std::string& bn_in_op =
        instr_ctx->operand_type(args->ibn(i)).Get<vm::StringObject>().str();
    int64_t index = args->input_index(i);
    const auto& blob_object = instr_ctx->operand_type(args->input_blob(i)).Get<BlobObject>();
    Callback(GenRepeatedBn(bn_in_op, index), blob_object);
  }
}

template<typename CallbackT>
void ForEachObnAndBlobObject(vm::InstrCtx* instr_ctx,
                             const FlatMsgView<CallOpKernelInstrOperand>& args,
                             const CallbackT& Callback) {
  CHECK_EQ(args->obn_size(), args->output_index_size());
  CHECK_EQ(args->obn_size(), args->output_blob_size());
  FOR_RANGE(int, i, 0, args->obn_size()) {
    const std::string& bn_in_op =
        instr_ctx->operand_type(args->obn(i)).Get<vm::StringObject>().str();
    int64_t index = args->output_index(i);
    auto* blob_object = instr_ctx->mut_operand_type(args->output_blob(i))->Mut<BlobObject>();
    Callback(GenRepeatedBn(bn_in_op, index), blob_object);
  }
  CHECK_EQ(args->mut2_obn_size(), args->mut2_output_index_size());
  CHECK_EQ(args->mut2_obn_size(), args->mut2_output_blob_size());
  FOR_RANGE(int, i, 0, args->mut2_obn_size()) {
    const std::string& bn_in_op =
        instr_ctx->operand_type(args->mut2_obn(i)).Get<vm::StringObject>().str();
    int64_t index = args->mut2_output_index(i);
    auto* blob_object = instr_ctx->mut_operand_type(args->mut2_output_blob(i))->Mut<BlobObject>();
    Callback(GenRepeatedBn(bn_in_op, index), blob_object);
  }
}

}  // namespace

void CallOpKernelInstructionType::Infer(vm::InstrCtx* instr_ctx) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instr_ctx->instr_msg().operand());
  HashMap<std::string, BlobDesc*> obn2blob_desc;
  {
    HashSet<const BlobDesc*> out_blob_descs;
    ForEachObnAndBlobObject(instr_ctx, args,
                            [&](const std::string& bn_in_op, BlobObject* blob_object) {
                              auto* blob_desc = blob_object->mut_blob_desc();
                              CHECK(out_blob_descs.insert(blob_desc).second);
                              CHECK(obn2blob_desc.emplace(bn_in_op, blob_desc).second);
                            });
  }
  HashMap<std::string, const BlobDesc*> ibn2blob_desc;
  ForEachIbnAndBlobObject(instr_ctx, args,
                          [&](const std::string& bn_in_op, const BlobObject& blob_object) {
                            CHECK(ibn2blob_desc.emplace(bn_in_op, &blob_object.blob_desc()).second);
                          });
  auto* opkernel_obj = instr_ctx->mut_operand_type(args->opkernel())->Mut<OpKernelObject>();
  auto BnInOp2BlobDesc = [&](const std::string& bn_in_op) -> BlobDesc* {
    auto output_iter = obn2blob_desc.find(bn_in_op);
    if (output_iter != obn2blob_desc.end()) { return output_iter->second; }
    auto input_iter = ibn2blob_desc.find(bn_in_op);
    if (input_iter != ibn2blob_desc.end()) { return const_cast<BlobDesc*>(input_iter->second); }
    return nullptr;
  };
  opkernel_obj->InferAndNewUninitiatedKernel(BnInOp2BlobDesc);
  ForEachObnAndBlobObject(instr_ctx, args, [](const std::string& _, BlobObject* blob_object) {
    blob_object->mutable_blob();
  });
}

void CallOpKernelInstructionType::Compute(vm::InstrCtx* instr_ctx) const {
  FlatMsgView<CallOpKernelInstrOperand> args(instr_ctx->instr_msg().operand());
  HashMap<std::string, Blob*> obn2blob;
  ForEachObnAndBlobObject(instr_ctx, args,
                          [&](const std::string& bn_in_op, BlobObject* blob_object) {
                            CHECK(obn2blob.emplace(bn_in_op, blob_object->mut_blob()).second);
                          });
  HashMap<std::string, const Blob*> ibn2blob;
  ForEachIbnAndBlobObject(instr_ctx, args,
                          [&](const std::string& bn_in_op, const BlobObject& blob_object) {
                            CHECK(ibn2blob.emplace(bn_in_op, &blob_object.blob()).second);
                          });
  auto* opkernel_obj = instr_ctx->mut_operand_type(args->opkernel())->Mut<OpKernelObject>();
  DeviceCtx* device_ctx = instr_ctx->mut_instr_chain()->stream().device_ctx().get();
  KernelCtx kernel_ctx;
  kernel_ctx.device_ctx = device_ctx;
  auto BnInOp2Blob = [&](const std::string& bn_in_op) -> Blob* {
    auto output_iter = obn2blob.find(bn_in_op);
    if (output_iter != obn2blob.end()) { return output_iter->second; }
    auto input_iter = ibn2blob.find(bn_in_op);
    if (input_iter != ibn2blob.end()) { return const_cast<Blob*>(input_iter->second); }
    return nullptr;
  };
  ForEachObnAndBlobObject(instr_ctx, args,
                          [&](const std::string& bn_in_op, BlobObject* blob_object) {
                            blob_object->TryAllocateBlobBodyMemory(device_ctx);
                          });
  opkernel_obj->kernel(kernel_ctx, BnInOp2Blob).Launch(kernel_ctx, BnInOp2Blob);
}

}  // namespace eager
}  // namespace oneflow
