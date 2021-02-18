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
#include <sstream>
#define private public
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/eager/transport_blob_instruction_type.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_util.h"

namespace oneflow {
namespace eager {
namespace test {

namespace {

struct SendRequest {
  int64_t dst_machine_id;
  uint64_t token;
  const char* mem_ptr;
  std::size_t size;
  std::function<void()> callback;
};

struct ReceiveRequest {
  int64_t src_machine_id;
  uint64_t token;
  char* mem_ptr;
  std::size_t size;
  std::function<void()> callback;
};

HashMap<uint64_t, SendRequest> token2send_request;
HashMap<uint64_t, ReceiveRequest> token2recv_request;

class TestSendBlobInstructionType : public SendBlobInstructionType {
 public:
  TestSendBlobInstructionType() = default;
  ~TestSendBlobInstructionType() override = default;

 private:
  Maybe<void> Send(int64_t dst_machine_id, uint64_t token, const char* mem_ptr, std::size_t size,
                   const std::function<void()>& Callback) const override {
    SendRequest send_request{
        .dst_machine_id = dst_machine_id,
        .token = token,
        .mem_ptr = mem_ptr,
        .size = size,
        .callback = Callback,
    };
    token2send_request.insert({token, send_request});
    const auto& iter = token2recv_request.find(token);
    if (iter != token2recv_request.end()) {
      const auto& recv_request = iter->second;
      CHECK_LE(send_request.size, recv_request.size);
      std::memcpy(recv_request.mem_ptr, send_request.mem_ptr, send_request.size);
      send_request.callback();
      recv_request.callback();
    }
    return Maybe<void>::Ok();
  }
};
COMMAND(vm::RegisterInstructionType<TestSendBlobInstructionType>("TestSendBlob"));

class TestReceiveBlobInstructionType : public ReceiveBlobInstructionType {
 public:
  TestReceiveBlobInstructionType() = default;
  ~TestReceiveBlobInstructionType() override = default;

 private:
  Maybe<void> Receive(int64_t src_machine_id, uint64_t token, char* mem_ptr, std::size_t size,
                      const std::function<void()>& Callback) const override {
    ReceiveRequest recv_request{
        .src_machine_id = src_machine_id,
        .token = token,
        .mem_ptr = mem_ptr,
        .size = size,
        .callback = Callback,
    };
    token2recv_request.insert({token, recv_request});
    const auto& iter = token2send_request.find(token);
    if (iter != token2send_request.end()) {
      const auto& send_request = iter->second;
      CHECK_LE(send_request.size, recv_request.size);
      std::memcpy(recv_request.mem_ptr, send_request.mem_ptr, send_request.size);
      send_request.callback();
      recv_request.callback();
    }
    return Maybe<void>::Ok();
  }
};
COMMAND(vm::RegisterInstructionType<TestReceiveBlobInstructionType>("TestReceiveBlob"));

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

int64_t NewJobDescSymbol(InstructionMsgList* list,
                         const std::shared_ptr<JobConfigProto>& job_conf) {
  int64_t job_desc_id = vm::TestUtil::NewSymbol(list);
  CHECK_JUST(Global<symbol::Storage<JobDesc>>::Get()->Add(job_desc_id, *job_conf));
  list->EmplaceBack(vm::NewInstruction("InitJobDescSymbol")->add_init_symbol_operand(job_desc_id));
  return job_desc_id;
}

int64_t NewOpConfSymbol(InstructionMsgList* list, const std::shared_ptr<OperatorConf>& op_conf) {
  int64_t op_conf_id = vm::TestUtil::NewSymbol(list);
  CHECK_JUST(Global<symbol::Storage<OperatorConfSymbol>>::Get()->Add(op_conf_id, *op_conf));
  list->EmplaceBack(
      vm::NewInstruction("InitOperatorConfSymbol")->add_init_symbol_operand(op_conf_id));
  return op_conf_id;
}

int64_t NewOpNodeSignature(InstructionMsgList* list, const std::vector<std::string>& ibns,
                           const std::vector<int64_t>& parallel_desc_symbol_ids_4_ibns,
                           const std::vector<std::string>& obns,
                           const std::vector<int64_t>& parallel_desc_symbol_ids_4_obns) {
  OpNodeSignature op_node_signature;
  const auto& SetFakeLogicalBlobDesc = [&](const std::string& bn_in_op) {
    auto* blob_sig = op_node_signature.mutable_logical_blob_desc_signature();
    auto* map = blob_sig->mutable_bn_in_op2blob_desc();
    BlobDesc(Shape({10LL}), DataType::kFloat).ToProto(&(*map)[bn_in_op]);
  };
  auto* bn_in_op2parallel_desc_symbol_id =
      op_node_signature.mutable_parallel_signature()->mutable_bn_in_op2parallel_desc_symbol_id();
  auto* map = op_node_signature.mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
  for (int i = 0; i < ibns.size(); ++i) {
    (*map)[ibns[i]].mutable_broadcast_parallel();
    (*bn_in_op2parallel_desc_symbol_id)[ibns[i]] = parallel_desc_symbol_ids_4_ibns[i];
    SetFakeLogicalBlobDesc(ibns[i]);
  }
  for (int i = 0; i < obns.size(); ++i) {
    (*map)[obns[i]].mutable_broadcast_parallel();
    (*bn_in_op2parallel_desc_symbol_id)[obns[i]] = parallel_desc_symbol_ids_4_obns[i];
    SetFakeLogicalBlobDesc(obns[i]);
  }
  int64_t op_node_signature_id = vm::TestUtil::NewSymbol(list);
  CHECK_JUST(Global<symbol::Storage<OpNodeSignatureDesc>>::Get()->Add(op_node_signature_id,
                                                                      op_node_signature));
  list->EmplaceBack(vm::NewInstruction("InitOpNodeSignatureDescSymbol")
                        ->add_init_symbol_operand(op_node_signature_id));
  return op_node_signature_id;
}

int64_t MakeTestBlob(InstructionMsgList* list, const std::string& parallel_str) {
  int64_t job_desc_id = NewJobDescSymbol(list, std::make_shared<JobConfigProto>());
  int64_t parallel_desc_id = 0;
  int64_t opkernel_id = vm::TestUtil::NewObject(list, "cpu", parallel_str, &parallel_desc_id);
  int64_t op_conf_id = 0;
  {
    auto op_conf = std::make_shared<OperatorConf>();
    op_conf->set_name("test_source_op_name");
    auto* user_conf = op_conf->mutable_user_conf();
    user_conf->set_op_type_name("TestSource");
    (*user_conf->mutable_output())["out"].add_s("test_source_op_name/out_0");
    op_conf_id = NewOpConfSymbol(list, op_conf);
  }
  int64_t op_node_signature_id = NewOpNodeSignature(list, {}, {}, {"out_0"}, {parallel_desc_id});
  int64_t obn_id = vm::TestUtil::NewStringSymbol(list, "out_0");
  int64_t output_blob_id = vm::TestUtil::NewObject(list, "cpu", parallel_str);
  list->EmplaceBack(vm::NewInstruction("cpu.compute.UserStatelessCallOpKernel")
                        ->add_parallel_desc(parallel_desc_id)
                        ->add_symbol_operand(job_desc_id)
                        ->add_symbol_operand(op_conf_id)
                        ->add_symbol_operand(op_node_signature_id)
                        ->add_mut_operand(opkernel_id)
                        ->add_separator()
                        ->add_separator()
                        ->add_separator()
                        ->add_symbol_operand(obn_id)
                        ->add_mut_operand(output_blob_id)
                        ->add_separator());
  return output_blob_id;
}

ObjectMsgPtr<vm::VirtualMachine> MakeVM(int64_t this_machine_id) {
  auto vm_desc =
      vm::MakeVmDesc(Global<ResourceDesc, ForSession>::Get()->resource(), this_machine_id,
                     {"NewObject", "InitJobDescSymbol", "InitOperatorConfSymbol",
                      "cpu.compute.UserStatelessCallOpKernel", "TestSendBlob", "TestReceiveBlob",
                      "SendBlob", "ReceiveBlob", "BroadcastObjectReference"});
  return ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
}

class SendRecvUtil {
 public:
  SendRecvUtil(const std::string& send_instr_name, const std::string& recv_instr_name)
      : send_instr_name_(send_instr_name), recv_instr_name_(recv_instr_name) {}
  ~SendRecvUtil() = default;

  void MakeTestInstructions(InstructionMsgList* list, uint64_t header_token, uint64_t body_token,
                            const std::string& src_pd, int64_t src_blob_id,
                            const std::string& dst_pd, int64_t dst_blob_id) const {
    int64_t src_pd_id = vm::TestUtil::NewParallelDesc(list, "cpu", src_pd);
    int64_t dst_pd_id = vm::TestUtil::NewParallelDesc(list, "cpu", dst_pd);
    list->EmplaceBack(vm::NewInstruction(send_instr_name_)
                          ->add_parallel_desc(src_pd_id)
                          ->add_symbol_operand(dst_pd_id)
                          ->add_const_operand(src_blob_id)
                          ->add_separator()
                          ->add_uint64_operand(header_token)
                          ->add_separator()
                          ->add_uint64_operand(body_token));
    list->EmplaceBack(vm::NewInstruction(recv_instr_name_)
                          ->add_parallel_desc(dst_pd_id)
                          ->add_symbol_operand(src_pd_id)
                          ->add_mut2_operand(dst_blob_id)
                          ->add_separator()
                          ->add_uint64_operand(header_token)
                          ->add_separator()
                          ->add_uint64_operand(body_token));
  }

 private:
  std::string send_instr_name_;
  std::string recv_instr_name_;
};

}  // namespace

TEST(SendReceiveInstructionType, naive) {
  vm::TestResourceDescScope scope(1, 1, 2);
  auto vm0 = MakeVM(0);
  int64_t src_blob_id = 0;
  {
    InstructionMsgList list;
    src_blob_id = MakeTestBlob(&list, "0:0");
    vm0->Receive(&list);
  }
  auto vm1 = MakeVM(1);
  int64_t dst_blob_id = 0;
  {
    InstructionMsgList list;
    dst_blob_id = MakeTestBlob(&list, "1:0");
    vm1->Receive(&list);
  }
  uint64_t header_token = 7777;
  uint64_t body_token = 8888;
  SendRecvUtil send_recv_util("TestSendBlob", "TestReceiveBlob");
  {
    InstructionMsgList list;
    send_recv_util.MakeTestInstructions(&list, header_token, body_token, "0:0", src_blob_id, "1:0",
                                        dst_blob_id);
    vm0->Receive(&list);
  }
  {
    InstructionMsgList list;
    send_recv_util.MakeTestInstructions(&list, header_token, body_token, "0:0", src_blob_id, "1:0",
                                        dst_blob_id);
    vm1->Receive(&list);
  }
  while (!(vm0->Empty() && vm1->Empty())) {
    vm0->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm0->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    vm1->Schedule();
    OBJECT_MSG_LIST_FOR_EACH_PTR(vm1->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  ASSERT_TRUE(token2send_request.find(header_token) != token2send_request.end());
  ASSERT_TRUE(token2recv_request.find(header_token) != token2recv_request.end());
  ASSERT_TRUE(token2send_request.find(body_token) != token2send_request.end());
  ASSERT_TRUE(token2recv_request.find(body_token) != token2recv_request.end());
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
