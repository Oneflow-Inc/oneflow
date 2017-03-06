#include "dag/pipe_dag.h"

namespace oneflow {

void PipeDag::Init(const StageDag* stage_dag,
                   const IDMap& id_map,
                   bool need_bp) {
  Stage2PnsMap stage2pns;
  InitComputePns(stage_dag, id_map, &stage2pns);
  InitBoxingPns(stage_dag, id_map, &stage2pns);
  ConnectPns(stage_dag, &stage2pns);
  if (need_bp) {
    GenerateBpNodes();
  }
  ConnectStartAndStop();
}

void PipeDag::InitComputePns(const StageDag* stage_dag,
                              const IDMap& id_map,
                              Stage2PnsMap* stage2pns) {
  for (const std::unique_ptr<DagNode>& node : stage_dag->node_vec()) {
    auto stage = of_dynamic_cast<const StageNode*> (node.get());
    bool is_first_stage = stage_dag->IsFirstNode(stage);
    bool is_last_stage = stage_dag->IsLastNode(stage);
    if (stage->parallel_desc().engine() == ParallelDesc::Engine::kDevice) {
      Stage2DeviceComputePns(stage,
                              id_map,
                              &((*stage2pns)[stage]),
                              is_first_stage,
                              is_last_stage);
    } else {
      Stage2HostComputePns(stage, id_map, &((*stage2pns)[stage]));
    }
  }
}

void PipeDag::Stage2DeviceComputePns(const StageNode* stage,
                                      const IDMap& id_map,
                                      PnsWithinStage* pns_within_stage,
                                      bool is_first_stage,
                                      bool is_last_stage) {
  MachineId machine_id = stage->machine_id();
  for (auto device_physical_id : stage->parallel_desc().devices_on_machine(machine_id)) {
    ThreadLocalId thread_local_id =
        id_map.ThreadLocalIdFromDevicePhysicalId(device_physical_id);
    // compute_pn
    DeviceComputePn* compute_pn = NewDeviceComputePn();
    compute_pn->mutable_layer_desc_vec() = stage->layer_desc_vec();
    compute_pn->mutable_parallel_desc_ptr() = stage->parallel_desc_ptr();
    compute_pn->mutable_machine_id() = machine_id;
    compute_pn->mutable_thread_local_id() = thread_local_id;
    // compute_in_pn
    if (!is_first_stage) {
      CopyHDPn* compute_in_pn = NewCopyHDPn();
      compute_in_pn->mutable_machine_id() = machine_id;
      compute_in_pn->mutable_thread_local_id() = thread_local_id;
      ConnectTwoNode(compute_in_pn, compute_pn);
      pns_within_stage->compute_in_pns.push_back(compute_in_pn);
    } else {
      pns_within_stage->compute_in_pns.push_back(compute_pn);
    }
    // compute_out_pn
    if (!is_last_stage) {
      CopyHDPn* compute_out_pn = NewCopyHDPn();
      compute_out_pn->mutable_machine_id() = machine_id;
      compute_out_pn->mutable_thread_local_id() = thread_local_id;
      ConnectTwoNode(compute_pn, compute_out_pn);
      pns_within_stage->compute_out_pns.push_back(compute_out_pn);
    } else {
      pns_within_stage->compute_out_pns.push_back(compute_pn);
    }
  }
}

void PipeDag::Stage2HostComputePns(const StageNode* stage,
                                    const IDMap& id_map,
                                    PnsWithinStage* pns_within_stage) {
  HostComputePn* compute_pn = NewHostComputePn();
  compute_pn->mutable_layer_desc_vec() = stage->layer_desc_vec();
  compute_pn->mutable_parallel_desc_ptr() = stage->parallel_desc_ptr();
  compute_pn->mutable_machine_id() = stage->machine_id();
  // since we only support GPU now, it must be a data-layer
  compute_pn->mutable_thread_local_id() = id_map.data_thread_local_id();
  pns_within_stage->compute_in_pns.push_back(compute_pn);
  pns_within_stage->compute_out_pns.push_back(compute_pn);
}

void PipeDag::InitBoxingPns(const StageDag* stage_dag,
                             const IDMap& id_map,
                             Stage2PnsMap* stage2pns) {
  for (const std::unique_ptr<DagNode>& node : stage_dag->node_vec()) {
    auto stage = of_dynamic_cast<const StageNode*> (node.get());
    InitInboxingPn(stage, id_map, &(stage2pns->at(stage)));
    InitOutBoxingPn(stage, id_map, &(stage2pns->at(stage)));
  }
}

void PipeDag::InitInboxingPn(const StageNode* stage,
                              const IDMap& id_map,
                              PnsWithinStage* pns_within_stage) {
  pns_within_stage->in_boxing_pn = nullptr;
  if (stage->predecessors().size() == 1
      && pns_within_stage->compute_in_pns.size() == 1) {
    return;
  }
  BoxingPn* boxing_pn = NewBoxingPn();
  boxing_pn->mutable_machine_id() = stage->machine_id();
  boxing_pn->mutable_thread_local_id() = id_map.boxing_thread_local_id();
  for (PipeNode* compute_in_pn : pns_within_stage->compute_in_pns) {
    ConnectTwoNode(boxing_pn, compute_in_pn);
  }
  pns_within_stage->in_boxing_pn = boxing_pn;
}

void PipeDag::InitOutBoxingPn(const StageNode* stage,
                               const IDMap& id_map,
                               PnsWithinStage* pns_within_stage) {
  pns_within_stage->out_boxing_pn = nullptr;
  if (stage->successors().size() == 1
      && pns_within_stage->compute_out_pns.size() == 1) {
    return;
  }
  BoxingPn* boxing_pn = NewBoxingPn();
  boxing_pn->mutable_machine_id() = stage->machine_id();
  boxing_pn->mutable_thread_local_id() = id_map.boxing_thread_local_id();
  for (PipeNode* compute_out_pn : pns_within_stage->compute_out_pns) {
    ConnectTwoNode(compute_out_pn, boxing_pn);
  }
  pns_within_stage->out_boxing_pn = boxing_pn;
}

void PipeDag::ConnectPns(const StageDag* stage_dag,
                          const Stage2PnsMap* stage2pns) {
  for (const std::unique_ptr<DagNode>& node : stage_dag->node_vec()) {
    auto cur_stage = of_dynamic_cast<const StageNode*> (node.get());
    const PnsWithinStage& cur_pns = stage2pns->at(cur_stage);
    PipeNode* out_node = cur_pns.out_boxing_pn;
    if (out_node == nullptr) {
      CHECK_EQ(cur_pns.compute_out_pns.size(), 1);
      out_node = cur_pns.compute_out_pns[0];
    }
    for (const DagNode* next : cur_stage->successors()) {
      auto next_stage = of_dynamic_cast<const StageNode*> (next);
      const PnsWithinStage& next_pns = stage2pns->at(next_stage);
      PipeNode* in_node = next_pns.in_boxing_pn;
      if (in_node == nullptr) {
        CHECK_EQ(next_pns.compute_in_pns.size(), 1);
        in_node = next_pns.compute_in_pns[0];
      }
      if (cur_stage->machine_id() == next_stage->machine_id()) {
        ConnectTwoNode(out_node, in_node);
      } else {
        CommNetPn* out_comm_net_node = NewCommNetPn();
        CommNetPn* in_comm_net_node = NewCommNetPn();
        ConnectTwoNode(out_node, out_comm_net_node);
        ConnectTwoNode(out_comm_net_node, in_comm_net_node);
        ConnectTwoNode(in_comm_net_node, in_node);
      }
    }
  }
}

void PipeDag::GenerateBpNodes() {
  // TODO
}

} // namespace oneflow
