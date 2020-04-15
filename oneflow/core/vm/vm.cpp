#include "oneflow/core/vm/vm.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace vm {

namespace {

bool IsSourceInstruction(const InstructionMsg& instr_msg) {
  for (const auto& instr_operand : instr_msg.operand()) {
    if (instr_operand->has_const_operand()) { return false; }
    if (instr_operand->has_mut_operand()) { return false; }
    if (instr_operand->has_mut2_operand()) { return false; }
    if (instr_operand->has_symbol_operand()) { return false; }
    if (instr_operand->has_init_symbol_operand()) { return false; }
    CHECK(instr_operand->has_separator() || instr_operand->has_double_operand()
          || instr_operand->has_double_operand() || instr_operand->has_int64_operand()
          || instr_operand->has_uint64_operand() || instr_operand->has_bool_operand());
  }
  return true;
}

}  // namespace

void VirtualMachine::ReleaseInstruction(InstrChain* instr_chain,
                                        /*out*/ ReadyInstrChainList* ready_instr_chain_list) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instr_ctx_list(), instr_ctx) {
    auto* mirrored_object_accesses = instr_ctx->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(mirrored_object_accesses, access) {
      mirrored_object_accesses->Erase(access);
      if (access->is_mirrored_object_access_link_empty()) { continue; }
      auto* mirrored_object = access->mut_mirrored_object();
      mirrored_object->mut_access_list()->Erase(access);
    }
  }
  auto* wait_instr_chain_list = mut_waiting_instr_chain_list();
  auto* out_edges = instr_chain->mut_out_edges();
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(out_edges, out_edge) {
    InstrChain* out_instr_chain = out_edge->dst_instr_chain();
    out_instr_chain->mut_in_edges()->Erase(out_edge);
    if (out_instr_chain->in_edges().empty()) {
      wait_instr_chain_list->MoveToDstBack(out_instr_chain, ready_instr_chain_list);
    }
    out_edges->Erase(out_edge);
  }
}

void VirtualMachine::TryReleaseFinishedInstrChains(
    Stream* stream,
    /*out*/ ReadyInstrChainList* ready_instr_chain_list) {
  auto* running_chain_list = stream->mut_running_chain_list();
  while (true) {
    auto* instr_chain_ptr = running_chain_list->Begin();
    if (instr_chain_ptr == nullptr || !instr_chain_ptr->Done()) { break; }
    ReleaseInstruction(instr_chain_ptr, /*out*/ ready_instr_chain_list);
    stream->DeleteInstrChain(running_chain_list->Erase(instr_chain_ptr));
  }
}

void VirtualMachine::FilterAndRunSourceInstructions(TmpPendingInstrMsgList* instr_msg_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(instr_msg_list, instr_msg) {
    const auto& instr_type_id = instr_msg->instr_type_id();
    const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
    if (stream_type.SharingVirtualMachineThread() && IsSourceInstruction(*instr_msg)) {
      stream_type.Run(this, instr_msg);
      instr_msg_list->Erase(instr_msg);
    }
  }
}

void VirtualMachine::MakeInstrChains(TmpPendingInstrMsgList* instr_msg_list,
                                     /*out*/ NewInstrChainList* new_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(instr_msg_list, instr_msg) {
    const StreamTypeId& stream_type_id = instr_msg->instr_type_id().stream_type_id();
    auto* stream_rt_desc = mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(stream_rt_desc->mut_stream_id2stream(), stream) {
      new_instr_chain_list->EmplaceBack(stream->NewInstrChain(instr_msg));
    }
    instr_msg_list->Erase(instr_msg);
  }
}

template<int64_t (*TransformLogicalObjectId)(int64_t), typename DoEachT>
void VirtualMachine::ForEachMirroredObject(Id2LogicalObject* id2logical_object,
                                           const Operand& operand, int64_t global_device_id,
                                           const DoEachT& DoEach) {
  int64_t logical_object_id = operand.logical_object_id();
  logical_object_id = TransformLogicalObjectId(logical_object_id);
  auto* logical_object = id2logical_object->FindPtr(logical_object_id);
  auto* map = logical_object->mut_global_device_id2mirrored_object();
  if (operand.has_all_mirrored_object()) {
    OBJECT_MSG_MAP_FOR_EACH_PTR(map, mirrored_object) { DoEach(mirrored_object); }
    return;
  }
  CHECK_NOTNULL(logical_object);
  auto* ret = map->FindPtr(operand.GetGlobalDeviceId(global_device_id));
  CHECK_NOTNULL(ret);
  DoEach(ret);
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachConstMirroredObject(
    InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kConstModifier, mem_zone_modifier>& const_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = const_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&GetTypeLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
    ForEachMirroredObject<&GetSelfLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&GetTypeLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachConstMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mutable_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&GetTypeLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachMutMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mutable_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&GetSelfLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&GetTypeLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachMutMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kTypeAndDataMutableModifier, mem_zone_modifier>& mut2_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mut2_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&GetTypeLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
    ForEachMirroredObject<&GetSelfLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&GetTypeLogicalObjectId>(id2logical_object, operand, global_device_id,
                                                   DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

void VirtualMachine::ConsumeMirroredObject(OperandAccessType access_type,
                                           MirroredObject* mirrored_object, InstrCtx* instr_ctx) {
  bool is_const_operand = (access_type == kConstOperandAccess);
  auto mirrored_object_access = ObjectMsgPtr<MirroredObjectAccess>::NewFrom(
      instr_ctx->mut_allocator(), instr_ctx, mirrored_object, is_const_operand);
  bool success =
      instr_ctx->mut_mirrored_object_id2access()->Insert(mirrored_object_access.Mutable()).second;
  if (success) {
    mirrored_object->mut_access_list()->EmplaceBack(std::move(mirrored_object_access));
  }
}

void VirtualMachine::ConnectInstruction(InstrChain* src_instr_chain, InstrChain* dst_instr_chain) {
  auto edge = ObjectMsgPtr<InstrChainEdge>::NewFrom(mut_vm_thread_only_allocator(), src_instr_chain,
                                                    dst_instr_chain);
  bool src_inserted = src_instr_chain->mut_out_edges()->Insert(edge.Mutable()).second;
  bool dst_inserted = dst_instr_chain->mut_in_edges()->Insert(edge.Mutable()).second;
  CHECK_EQ(src_inserted, dst_inserted);
}

void VirtualMachine::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                            NewInstrChainList* new_instr_chain_list) {
  auto* begin = new_instr_chain_list->Begin();
  if (begin != nullptr) { CHECK_EQ(begin->instr_ctx_list().size(), 1); }
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_instr_chain_list, instr_chain) {
    int64_t global_device_id = instr_chain->stream().global_device_id();
    InterpretType interpret_type = instr_chain->stream().stream_type_id().interpret_type();
    CHECK_EQ(instr_chain->instr_ctx_list().size(), 1);
    auto* instr_ctx = instr_chain->mut_instr_ctx_list()->Begin();
    auto ConsumeMutMirroredObject = [&](MirroredObject* mirrored_object) {
      ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, instr_ctx);
    };
    auto ConsumeConstMirroredObject = [&](MirroredObject* mirrored_object) {
      ConsumeMirroredObject(kConstOperandAccess, mirrored_object, instr_ctx);
    };
    const auto& operands = instr_ctx->instr_msg().operand();
    for (const auto& operand : operands) {
      if (operand->has_mut_operand()) {
        ForEachMutMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                         operand->mut_operand(), global_device_id,
                                                         ConsumeMutMirroredObject);
      } else if (operand->has_mut2_operand()) {
        ForEachMutMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                         operand->mut2_operand(), global_device_id,
                                                         ConsumeMutMirroredObject);
      } else if (operand->has_init_symbol_operand()) {
        const auto& symbol_operand = operand->init_symbol_operand().operand();
        CHECK(symbol_operand.has_sole_mirrored_object());
        ForEachMutMirroredObject<kHostConstMemZoneModifier>(interpret_type, id2logical_object,
                                                            operand->init_symbol_operand(), 0,
                                                            ConsumeMutMirroredObject);
      } else {
        // do nothing
      }
    }
    for (const auto& operand : operands) {
      if (operand->has_const_operand()) {
        ForEachConstMirroredObject<kDeviceMemZoneModifier>(
            interpret_type, id2logical_object, operand->const_operand(), global_device_id,
            ConsumeConstMirroredObject);
      } else if (operand->has_mut_operand()) {
        ForEachConstMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                           operand->mut_operand(), global_device_id,
                                                           ConsumeConstMirroredObject);
      } else if (operand->has_symbol_operand()) {
        const auto& symbol_operand = operand->symbol_operand().operand();
        CHECK(symbol_operand.has_sole_mirrored_object());
        ForEachConstMirroredObject<kHostConstMemZoneModifier>(interpret_type, id2logical_object,
                                                              operand->symbol_operand(), 0,
                                                              ConsumeConstMirroredObject);
      } else if (operand->has_init_symbol_operand()) {
        const auto& symbol_operand = operand->init_symbol_operand().operand();
        CHECK(symbol_operand.has_sole_mirrored_object());
        ForEachConstMirroredObject<kHostConstMemZoneModifier>(interpret_type, id2logical_object,
                                                              operand->init_symbol_operand(), 0,
                                                              ConsumeConstMirroredObject);
      } else {
        // do nothing
      }
    }
    auto* mirrored_object_accesses = instr_ctx->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(mirrored_object_accesses, mirrored_object_access) {
      auto* mirrored_object = mirrored_object_access->mut_mirrored_object();
      if (mirrored_object->access_list().size() == 1) { continue; }
      if (mirrored_object_access->is_const_operand()) {
        auto* first = mirrored_object->mut_access_list()->Begin();
        if (!first->is_const_operand()) {
          ConnectInstruction(first->mut_instr_ctx()->mut_instr_chain(), instr_chain);
        }
      } else {
        auto* access_list = mirrored_object->mut_access_list();
        OBJECT_MSG_LIST_FOR_EACH_PTR(access_list, access) {
          if (access == mirrored_object_access) { break; }
          ConnectInstruction(access->mut_instr_ctx()->mut_instr_chain(), instr_chain);
          access_list->Erase(access);
        }
      }
    }
  }
}

void VirtualMachine::MergeChains(NewInstrChainList* new_instr_chain_list) {
  // TODO(lixinqi)
}

void VirtualMachine::FilterReadyChains(NewInstrChainList* new_instr_chain_list,
                                       /*out*/ ReadyInstrChainList* ready_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_instr_chain_list, instr_chain) {
    if (instr_chain->in_edges().empty()) {
      new_instr_chain_list->MoveToDstBack(instr_chain, ready_instr_chain_list);
    }
  }
}

void VirtualMachine::DispatchInstruction(ReadyInstrChainList* ready_chain_list) {
  auto* active_stream_list = mut_active_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(ready_chain_list, instr_chain) {
    auto* stream = instr_chain->mut_stream();
    ready_chain_list->MoveToDstBack(instr_chain, stream->mut_running_chain_list());
    if (stream->is_active_stream_link_empty()) { active_stream_list->PushBack(stream); }
    const auto& stream_type = stream->stream_type();
    if (stream_type.SharingVirtualMachineThread()) {
      stream_type.Run(this, instr_chain);
    } else {
      stream->mut_thread_ctx()->mut_pending_chain_list()->PushBack(instr_chain);
    }
  }
  ready_chain_list->Clear();
}

void VirtualMachine::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  mutable_vm_resource_desc()->CopyFrom(vm_desc.vm_resource_desc());
  CHECK_GT(vm_desc.machine_id_range().size(), 0);
  *mutable_machine_id_range() = vm_desc.machine_id_range();
  set_vm_thread_only_allocator(allocator);
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(&vm_desc.stream_type_id2desc(), stream_desc) {
    auto stream_rt_desc = ObjectMsgPtr<StreamRtDesc>::NewFrom(allocator, stream_desc);
    mut_stream_type_id2stream_rt_desc()->Insert(stream_rt_desc.Mutable());
    BalancedSplitter bs(stream_desc->parallel_num(), stream_desc->num_threads());
    for (int64_t i = 0, rel_global_device_id = 0; i < stream_desc->num_threads(); ++i) {
      auto thread_ctx = ObjectMsgPtr<ThreadCtx>::NewFrom(allocator, stream_rt_desc.Get(), i);
      mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      for (int j = bs.At(i).begin(); j < bs.At(i).end(); ++j, ++rel_global_device_id) {
        StreamId stream_id;
        stream_id.__Init__(stream_desc->stream_type_id(),
                           stream_desc->start_global_device_id() + rel_global_device_id);
        auto stream =
            ObjectMsgPtr<Stream>::NewFrom(mut_allocator(), thread_ctx.Mutable(), stream_id);
        CHECK(stream_rt_desc->mut_stream_id2stream()->Insert(stream.Mutable()).second);
        thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
      }
    }
  }
}

void VirtualMachine::Receive(InstructionMsgList* compute_instr_msg_list) {
  InstructionMsgList new_instr_msg_list;
  OBJECT_MSG_LIST_FOR_EACH_PTR(compute_instr_msg_list, compute_instr_msg) {
    new_instr_msg_list.EmplaceBack(compute_instr_msg->MakeInferInstrMsg());
    compute_instr_msg_list->MoveToDstBack(compute_instr_msg, &new_instr_msg_list);
  }
  mut_pending_msg_list()->MoveFrom(&new_instr_msg_list);
}

void VirtualMachine::Receive(ObjectMsgPtr<InstructionMsg>&& compute_instr_msg) {
  InstructionMsgList instr_msg_list;
  instr_msg_list.EmplaceBack(std::move(compute_instr_msg));
  Receive(&instr_msg_list);
}

void VirtualMachine::Schedule() {
  ReadyInstrChainList ready_instr_chain_list;
  auto* active_stream_list = mut_active_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_stream_list, stream) {
    TryReleaseFinishedInstrChains(stream, /*out*/ &ready_instr_chain_list);
    if (stream->running_chain_list().empty()) { active_stream_list->Erase(stream); }
  };
  auto* waiting_instr_chain_list = mut_waiting_instr_chain_list();
  if (pending_msg_list().size() > 0) {
    TmpPendingInstrMsgList tmp_pending_msg_list;
    mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
    FilterAndRunSourceInstructions(&tmp_pending_msg_list);
    NewInstrChainList new_instr_chain_list;
    MakeInstrChains(&tmp_pending_msg_list, /*out*/ &new_instr_chain_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_instr_chain_list);
    MergeChains(&new_instr_chain_list);
    FilterReadyChains(&new_instr_chain_list, /*out*/ &ready_instr_chain_list);
    new_instr_chain_list.MoveTo(waiting_instr_chain_list);
  }
  DispatchInstruction(&ready_instr_chain_list);
}

bool VirtualMachine::Empty() const {
  return pending_msg_list().empty() && waiting_instr_chain_list().empty()
         && active_stream_list().empty();
}

}  // namespace vm
}  // namespace oneflow
