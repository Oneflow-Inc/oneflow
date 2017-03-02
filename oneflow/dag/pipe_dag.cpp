#include "dag/pipe_dag.h"

namespace oneflow {

void PipeDag::Init(std::shared_ptr<const StageDag> stage_dag,
                   const IDMap& id_map) {
  Stage2PonsMap stage2pons;
  InitComputePons(stage_dag.get(), id_map, &stage2pons);
  InitBoxingPons(stage_dag.get(), id_map, &stage2pons);
  ConnectPons(stage_dag.get(), &stage2pons);
}

void PipeDag::InitComputePons(const StageDag* stage_dag,
                              const IDMap& id_map,
                              Stage2PonsMap* stage2pons) {
  for (const std::unique_ptr<OpNode>& opnode : stage_dag->op_node_vec()) {
    auto stage_op = of_dynamic_cast<const StageOpNode*> (opnode.get());
    bool is_first_stage = stage_dag->IsFirstNode(stage_op);
    bool is_last_stage = stage_dag->IsLastNode(stage_op);
    if (stage_op->parallel_desc().engine() == ParallelDesc::Engine::kDevice) {
      Stage2DeviceComputePons(stage_op,
                              id_map,
                              &((*stage2pons)[stage_op]),
                              is_first_stage,
                              is_last_stage);
    } else {
      Stage2HostComputePons(stage_op, id_map, &((*stage2pons)[stage_op]));
    }
  }
}

void PipeDag::Stage2DeviceComputePons(const StageOpNode* stage_op,
                                      const IDMap& id_map,
                                      PonsWithinStage* pons_within_stage,
                                      bool is_first_stage,
                                      bool is_last_stage) {
  MachineId machine_id = stage_op->machine_id();
  for (auto device_physical_id : stage_op->parallel_desc().devices_on_machine(machine_id)) {
    ThreadLocalId thread_local_id =
        id_map.ThreadLocalIdFromDevicePhysicalId(device_physical_id);
    // compute_pon
    DeviceComputePon* compute_pon = NewDeviceComputePon();
    compute_pon->mutable_layer_desc_vec() = stage_op->layer_desc_vec();
    compute_pon->mutable_parallel_desc_ptr() = stage_op->parallel_desc_ptr();
    compute_pon->mutable_machine_id() = machine_id;
    compute_pon->mutable_thread_local_id() = thread_local_id;
    // compute_in_pon
    if (!is_first_stage) {
      CopyHDPon* compute_in_pon = NewCopyHDPon();
      compute_in_pon->mutable_machine_id() = machine_id;
      compute_in_pon->mutable_thread_local_id() = thread_local_id;
      ConnectTwoOp(compute_in_pon, compute_pon);
      pons_within_stage->compute_in_pons.push_back(compute_in_pon);
    } else {
      pons_within_stage->compute_in_pons.push_back(compute_pon);
    }
    // compute_out_pon
    if (!is_last_stage) {
      CopyHDPon* compute_out_pon = NewCopyHDPon();
      compute_out_pon->mutable_machine_id() = machine_id;
      compute_out_pon->mutable_thread_local_id() = thread_local_id;
      ConnectTwoOp(compute_pon, compute_out_pon);
      pons_within_stage->compute_out_pons.push_back(compute_out_pon);
    } else {
      pons_within_stage->compute_out_pons.push_back(compute_pon);
    }
  }
}

void PipeDag::Stage2HostComputePons(const StageOpNode* stage_op,
                                    const IDMap& id_map,
                                    PonsWithinStage* pons_within_stage) {
  HostComputePon* compute_pon = NewHostComputePon();
  compute_pon->mutable_layer_desc_vec() = stage_op->layer_desc_vec();
  compute_pon->mutable_parallel_desc_ptr() = stage_op->parallel_desc_ptr();
  compute_pon->mutable_machine_id() = stage_op->machine_id();
  // since we only support GPU now, it must be a data-layer
  compute_pon->mutable_thread_local_id() = id_map.data_thread_local_id();
  pons_within_stage->compute_in_pons.push_back(compute_pon);
  pons_within_stage->compute_out_pons.push_back(compute_pon);
}

void PipeDag::InitBoxingPons(const StageDag* stage_dag,
                             const IDMap& id_map,
                             Stage2PonsMap* stage2pons) {
  for (const std::unique_ptr<OpNode>& opnode : stage_dag->op_node_vec()) {
    auto stage_op = of_dynamic_cast<const StageOpNode*> (opnode.get());
    InitInboxingPon(stage_op, id_map, &(stage2pons->at(stage_op)));
    InitOutBoxingPon(stage_op, id_map, &(stage2pons->at(stage_op)));
  }
}

void PipeDag::InitInboxingPon(const StageOpNode* stage_op,
                              const IDMap& id_map,
                              PonsWithinStage* pons_within_stage) {
  pons_within_stage->in_boxing_pon = nullptr;
  if (stage_op->predecessors().size() == 1
      && pons_within_stage->compute_in_pons.size() == 1) {
    return;
  }
  BoxingPon* boxing_pon = NewBoxingPon();
  boxing_pon->mutable_machine_id() = stage_op->machine_id();
  boxing_pon->mutable_thread_local_id() = id_map.boxing_thread_local_id();
  for (PipeOpNode* compute_in_pon : pons_within_stage->compute_in_pons) {
    ConnectTwoOp(boxing_pon, compute_in_pon);
  }
  pons_within_stage->in_boxing_pon = boxing_pon;
}

void PipeDag::InitOutBoxingPon(const StageOpNode* stage_op,
                               const IDMap& id_map,
                               PonsWithinStage* pons_within_stage) {
  pons_within_stage->out_boxing_pon = nullptr;
  if (stage_op->successors().size() == 1
      && pons_within_stage->compute_out_pons.size() == 1) {
    return;
  }
  BoxingPon* boxing_pon = NewBoxingPon();
  boxing_pon->mutable_machine_id() = stage_op->machine_id();
  boxing_pon->mutable_thread_local_id() = id_map.boxing_thread_local_id();
  for (PipeOpNode* compute_out_pon : pons_within_stage->compute_out_pons) {
    ConnectTwoOp(compute_out_pon, boxing_pon);
  }
  pons_within_stage->out_boxing_pon = boxing_pon;
}

void PipeDag::ConnectPons(const StageDag* stage_dag,
                          const Stage2PonsMap* stage2pons) {
  for (const std::unique_ptr<OpNode>& opnode : stage_dag->op_node_vec()) {
    auto cur_stage_op = of_dynamic_cast<const StageOpNode*> (opnode.get());
    const PonsWithinStage& cur_pons = stage2pons->at(cur_stage_op);
    PipeOpNode* out_node = cur_pons.out_boxing_pon;
    if (out_node == nullptr) {
      CHECK_EQ(cur_pons.compute_out_pons.size(), 1);
      out_node = cur_pons.compute_out_pons[0];
    }
    for (const OpNode* next_op : cur_stage_op->op_successors()) {
      auto next_stage_op = of_dynamic_cast<const StageOpNode*> (next_op);
      const PonsWithinStage& next_pons = stage2pons->at(next_stage_op);
      PipeOpNode* in_node = next_pons.in_boxing_pon;
      if (in_node == nullptr) {
        CHECK_EQ(next_pons.compute_in_pons.size(), 1);
        in_node = next_pons.compute_in_pons[0];
      }
      if (cur_stage_op->machine_id() == next_stage_op->machine_id()) {
        ConnectTwoOp(out_node, in_node);
      } else {
        CommNetPon* out_comm_net_node = NewCommNetPon();
        CommNetPon* in_comm_net_node = NewCommNetPon();
        ConnectTwoOp(out_node, out_comm_net_node);
        ConnectTwoOp(out_comm_net_node, in_comm_net_node);
        ConnectTwoOp(in_comm_net_node, in_node);
      }
    }
  }
}

} // namespace oneflow
