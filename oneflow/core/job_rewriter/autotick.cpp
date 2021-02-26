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
#include "oneflow/core/job_rewriter/autotick.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace {

std::unique_ptr<MutOpConTickInputHelper> NewMutOpConTickInputHelper(const OperatorConf& op_conf) {
  std::unique_ptr<MutOpConTickInputHelper> ret;
  if (IsClassRegistered<int32_t, MutOpConTickInputHelper>(op_conf.op_type_case())) {
    ret.reset(NewObj<int32_t, MutOpConTickInputHelper>(op_conf.op_type_case()));
    ret->InitFromOpConf(op_conf);
  }
  return ret;
}

void PrependTickByParallelDesc(const OpGraph& op_graph, JobBuilder* job_builder) {
  HashMap<ParallelDesc, std::vector<OpNode*>> parallel_desc2op_node;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
    if (!mut_tick_input_helper) { return; }
    if (mut_tick_input_helper->IsTickInputBound() == true) { return; }
    parallel_desc2op_node[op_node->parallel_desc()].push_back(op_node);
  });
  for (const auto& pair : parallel_desc2op_node) {
    OperatorConf device_tick_op;
    device_tick_op.set_name("System-AutoTick-Prepend-DeviceTick_" + NewUniqueId());
    auto* device_tick_op_conf = device_tick_op.mutable_device_tick_conf();
    device_tick_op_conf->set_out("out");
    job_builder->AddOps(pair.first.parallel_conf(), {device_tick_op});

    for (const auto* op_node : pair.second) {
      auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
      job_builder->MutOpsOnlyOnce(
          {mut_tick_input_helper->NewTickInputBoundOpConf(device_tick_op.name() + "/out")});
    }
  }
}

Maybe<const OperatorConf&> FindSrcSubsetTickOpConf(const Job& job) {
  const OperatorConf* src_subset_tick_op_conf = nullptr;
  for (const auto& op_conf : job.net().op()) {
    if (!op_conf.has_src_subset_tick_conf()) { continue; }
    CHECK_ISNULL_OR_RETURN(src_subset_tick_op_conf);
    src_subset_tick_op_conf = &op_conf;
  }
  CHECK_NOTNULL_OR_RETURN(src_subset_tick_op_conf);
  return *src_subset_tick_op_conf;
}

Maybe<void> BuildDstSubsetTickOpAndParallelConf(const HashSet<LogicalBlobId>& tick_lbis,
                                                OperatorConf* dst_subset_tick_op,
                                                JobBuilder* job_builder) {
  dst_subset_tick_op->set_name("System-AutoTick-DstSubsetTick_" + NewUniqueId());
  auto* dst_subset_tick_op_conf = dst_subset_tick_op->mutable_dst_subset_tick_conf();
  dst_subset_tick_op_conf->set_out("out");
  for (const LogicalBlobId& tick_lbi : tick_lbis) {
    dst_subset_tick_op_conf->add_in(GenLogicalBlobName(tick_lbi));
  }
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  int64_t num_machines = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  for (int64_t machine_id = 0; machine_id < num_machines; ++machine_id) {
    parallel_conf.add_device_name(std::to_string(machine_id) + ":0");
  }
  JUST(job_builder->AddOp(parallel_conf, *dst_subset_tick_op));
  return Maybe<void>::Ok();
}

Maybe<void> CreateDstSubsetTickAndSinkTicks(CriticalSection* critical_section,
                                            const OperatorConf& src_subset_tick,
                                            const HashSet<LogicalBlobId>& tick_lbis,
                                            JobBuilder* job_builder) {
  OperatorConf dst_subset_tick;
  dst_subset_tick.mutable_dst_subset_tick_conf()->add_in(
      src_subset_tick.name() + "/" + src_subset_tick.src_subset_tick_conf().out());
  JUST(BuildDstSubsetTickOpAndParallelConf(tick_lbis, &dst_subset_tick, job_builder));
  int64_t num_machines = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  auto* map = critical_section->mutable_machine_id2sink_tick_op_name();
  for (int64_t machine_id = 0; machine_id < num_machines; ++machine_id) {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::to_string(machine_id) + ":0");
    OperatorConf tick_op;
    {
      tick_op.set_name("System-AutoTick-Tick_" + NewUniqueId());
      auto* tick_conf = tick_op.mutable_tick_conf();
      tick_conf->add_tick(dst_subset_tick.name() + "/"
                          + dst_subset_tick.dst_subset_tick_conf().out());
      tick_conf->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, tick_op));
    }
    OperatorConf sink_tick_op;
    {
      sink_tick_op.set_name("System-AutoTick-SinkTick_" + NewUniqueId());
      auto* sink_tick_conf = sink_tick_op.mutable_sink_tick_conf();
      sink_tick_conf->add_tick(tick_op.name() + "/out");
      sink_tick_conf->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, sink_tick_op));
    }
    (*map)[machine_id] = sink_tick_op.name();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BuildSrcSubsetTickOpAndParallelConf(OperatorConf* src_subset_tick_op,
                                                JobBuilder* job_builder) {
  src_subset_tick_op->set_name("System-AutoTick-SrcSubsetTick_" + NewUniqueId());
  src_subset_tick_op->mutable_src_subset_tick_conf()->set_out("out");
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  int64_t num_machines = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  for (int64_t machine_id = 0; machine_id < num_machines; ++machine_id) {
    parallel_conf.add_device_name(std::to_string(machine_id) + ":0");
  }
  JUST(job_builder->AddOp(parallel_conf, *src_subset_tick_op));
  return Maybe<void>::Ok();
}

Maybe<void> CreateSourceTicksAndSrcSubsetTick(CriticalSection* critical_section,
                                              OperatorConf* src_subset_tick_op,
                                              JobBuilder* job_builder) {
  int64_t num_machines = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  auto* map = critical_section->mutable_machine_id2source_tick_op_name();
  for (int64_t machine_id = 0; machine_id < num_machines; ++machine_id) {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::to_string(machine_id) + ":0");
    OperatorConf src_tick_op;
    {
      src_tick_op.set_name("System-AutoTick-SourceTick_" + NewUniqueId());
      src_tick_op.mutable_source_tick_conf()->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, src_tick_op));
    }
    (*map)[machine_id] = src_tick_op.name();
    OperatorConf tick_op;
    {
      tick_op.set_name("System-AutoTick-Tick_" + NewUniqueId());
      tick_op.mutable_tick_conf()->add_tick(src_tick_op.name() + "/out");
      tick_op.mutable_tick_conf()->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, tick_op));
    }
    src_subset_tick_op->mutable_src_subset_tick_conf()->add_in(tick_op.name() + "/out");
  }
  JUST(job_builder->MutOpOnlyOnce(*src_subset_tick_op));
  return Maybe<void>::Ok();
}

Maybe<void> ConnectSrcSubsetTickAndOtherTick(const OperatorConf& src_subset_tick_op,
                                             JobBuilder* job_builder) {
  CHECK(src_subset_tick_op.has_src_subset_tick_conf());
  const std::string& src_lbn =
      src_subset_tick_op.name() + "/" + src_subset_tick_op.src_subset_tick_conf().out();
  job_builder->ForEachOperator([&](const Operator& op) {
    if (op.op_name() != src_subset_tick_op.name()) {
      CHECK(!op.op_conf().has_src_subset_tick_conf());
    }
    auto mut_helper = NewMutOpConTickInputHelper(op.op_conf());
    if (!mut_helper) { return; }
    if (mut_helper->IsTickInputBound() == true) { return; }
    job_builder->MutOpsOnlyOnce({mut_helper->NewTickInputBoundOpConf(src_lbn)});
  });
  return Maybe<void>::Ok();
}

const OpNode* GetSrcSubsetTickOpNode(const OpGraph& op_graph) {
  const OpNode* src_subset_tick = nullptr;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_src_subset_tick_conf()) {
      CHECK_ISNULL(src_subset_tick);
      src_subset_tick = op_node;
    }
  });
  CHECK_NOTNULL(src_subset_tick);
  return src_subset_tick;
}

const Shape& GetOpTimeShape(const OpNode* op_node) {
  const Shape* output_shape = op_node->out_blob_time_shape();
  if (output_shape == nullptr) { output_shape = op_node->GetInputBlobFastestTimeShape(); }
  return *output_shape;
};

OperatorConf MakeTickOpConf(const std::string& tick_name) {
  OperatorConf tick_op_conf;
  tick_op_conf.set_name(std::string("System-AutoTick-" + tick_name + "Tick_") + NewUniqueId());
  auto* tick_conf = tick_op_conf.mutable_tick_conf();
  tick_conf->set_out("out");
  return tick_op_conf;
}

OperatorConf MakeDeviceTickOpConf(const std::string& tick_name) {
  OperatorConf device_tick_op_conf;
  device_tick_op_conf.set_name(std::string("System-AutoTick-" + tick_name + "DeviceTick_")
                               + NewUniqueId());
  auto* tick_conf = device_tick_op_conf.mutable_device_tick_conf();
  tick_conf->set_out("out");
  return device_tick_op_conf;
}

OperatorConf AppendTick(const std::string tick_name, const std::vector<std::string>& op_names,
                        ParallelConf parallel_conf, JobBuilder* job_builder) {
  OperatorConf device_tick_op_conf = MakeDeviceTickOpConf(tick_name);
  for (const auto& op_name : op_names) { device_tick_op_conf.add_ctrl_in_op_name(op_name); }
  job_builder->AddOps(parallel_conf, {device_tick_op_conf});
  return device_tick_op_conf;
}

OperatorConf AppendTick(const std::string tick_name, const std::list<const OpNode*>& op_nodes,
                        JobBuilder* job_builder) {
  std::vector<std::string> op_names;
  for (const auto* op_node : op_nodes) {
    CHECK(op_nodes.front()->parallel_desc() == op_node->parallel_desc());
    op_names.push_back(op_node->op().op_name());
  }
  return AppendTick(tick_name, op_names, op_nodes.front()->parallel_desc().parallel_conf(),
                    job_builder);
}

OperatorConf PrependTick(const HashSet<const OpNode*>& op_nodes, JobBuilder* job_builder) {
  CHECK_GE(op_nodes.size(), 1);
  OperatorConf tick_op_conf = MakeTickOpConf("Prepend");
  std::vector<OperatorConf> op_confs;
  for (const OpNode* op_node : op_nodes) {
    OperatorConf op_conf(op_node->op().op_conf());
    op_conf.add_ctrl_in_op_name(tick_op_conf.name());
    op_confs.push_back(op_conf);
  }
  job_builder->MutOpsOnlyOnce({op_confs});
  ParallelDesc pd((*op_nodes.begin())->parallel_desc());
  pd.set_device_type(DeviceType::kCPU);
  job_builder->AddOps(pd.parallel_conf(), {tick_op_conf});
  return tick_op_conf;
}

OperatorConf AppendAccTick(const Shape& src_shape, const std::list<const OpNode*>& op_nodes,
                           JobBuilder* job_builder) {
  const auto& tick_shape = GetOpTimeShape(op_nodes.front());
  CHECK_EQ(tick_shape.elem_cnt() % src_shape.elem_cnt(), 0);
  const OperatorConf& tick_op_conf = AppendTick("AppendAcc", op_nodes, job_builder);
  OperatorConf acc_op_conf;
  {
    acc_op_conf.set_name(std::string("System-AutoTick-AccTick_") + NewUniqueId());
    auto* acc_conf = acc_op_conf.mutable_acc_tick_conf();
    CHECK(tick_op_conf.has_device_tick_conf());
    acc_conf->set_one(tick_op_conf.name() + "/" + tick_op_conf.device_tick_conf().out());
    acc_conf->set_acc("acc");
    acc_conf->set_max_acc_num(tick_shape.elem_cnt() / src_shape.elem_cnt());
  }
  OperatorConf last_device_tick_op_conf;
  {
    last_device_tick_op_conf.set_name(std::string("System-AutoTick-Tick_") + NewUniqueId());
    auto* device_tick_conf = last_device_tick_op_conf.mutable_device_tick_conf();
    device_tick_conf->add_tick(acc_op_conf.name() + "/acc");
    device_tick_conf->set_out("out");
  }
  job_builder->AddOps(op_nodes.front()->parallel_desc().parallel_conf(),
                      {acc_op_conf, last_device_tick_op_conf});
  return last_device_tick_op_conf;
}

std::vector<std::string> GetOpNames(const HashSet<const OpNode*>& op_nodes) {
  std::vector<std::string> ret;
  for (const OpNode* op_node : op_nodes) { ret.push_back(op_node->op().op_name()); }
  return ret;
};

void InitOpTypeCase2OpNodes(
    const OpGraph& op_graph,
    HashMap<OperatorConf::OpTypeCase, HashSet<const OpNode*>>* op_type_case2op_nodes) {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (IsInterfaceOpConf(op_conf)) {
      CHECK((*op_type_case2op_nodes)[op_conf.op_type_case()].emplace(op_node).second);
    }
  });
}

void ForEachInputCriticalSectionOpNodes(
    const OpGraph& op_graph,
    const std::function<void(const HashSet<const OpNode*>&, const std::vector<std::string>&)>&
        Handler) {
  HashMap<OperatorConf::OpTypeCase, HashSet<const OpNode*>> op_type_case2op_nodes;
  InitOpTypeCase2OpNodes(op_graph, &op_type_case2op_nodes);
  OperatorConf::OpTypeCase op_type_case = OperatorConf::kInputConf;
  if (op_type_case2op_nodes[op_type_case].empty()) { return; }
  HashSet<const OpNode*> op_nodes = op_type_case2op_nodes[op_type_case];
  for (const OpNode* op_node : op_type_case2op_nodes[op_type_case]) {
    op_node->ForEachNodeOnOutEdge([&](OpNode* out_node) { op_nodes.insert(out_node); });
  }
  Handler(op_nodes, GetOpNames(op_type_case2op_nodes[op_type_case]));
}

void ForEachOutputCriticalSectionOpNodes(
    const OpGraph& op_graph,
    const std::function<void(const HashSet<const OpNode*>&, const std::vector<std::string>&)>&
        Handler) {
  HashMap<OperatorConf::OpTypeCase, HashSet<const OpNode*>> op_type_case2op_nodes;
  InitOpTypeCase2OpNodes(op_graph, &op_type_case2op_nodes);
  if (op_type_case2op_nodes[OperatorConf::kReturnConf].empty() == false) {
    Handler(op_type_case2op_nodes[OperatorConf::kReturnConf],
            GetOpNames(op_type_case2op_nodes[OperatorConf::kReturnConf]));
  }
  if (op_type_case2op_nodes[OperatorConf::kOutputConf].empty() == false) {
    Handler(op_type_case2op_nodes[OperatorConf::kOutputConf],
            GetOpNames(op_type_case2op_nodes[OperatorConf::kOutputConf]));
  }
}

std::vector<OperatorConf> AddTickForTimeShape(const Shape& src_time_shape,
                                              const HashSet<const OpNode*>& op_nodes,
                                              JobBuilder* job_builder) {
  HashMap<std::pair<ParallelDesc, std::pair<Shape, Shape>>, std::list<const OpNode*>>
      pd7ts2op_nodes;
  for (const OpNode* op_node : op_nodes) {
    auto ts = std::make_pair(*op_node->GetInputOutputFastestTimeShape(), GetOpTimeShape(op_node));
    pd7ts2op_nodes[{op_node->parallel_desc(), ts}].push_back(op_node);
  }
  std::vector<OperatorConf> op_confs;
  for (const auto& pair : pd7ts2op_nodes) {
    const std::pair<Shape, Shape>& ts = pair.first.second;
    if (ts.second.elem_cnt() == src_time_shape.elem_cnt()) {
      CHECK_GE(ts.first.elem_cnt(), ts.second.elem_cnt());
      op_confs.push_back(AppendTick("Append", pair.second, job_builder));
    } else if (ts.second.elem_cnt() > src_time_shape.elem_cnt()) {
      op_confs.push_back(AppendAccTick(src_time_shape, pair.second, job_builder));
    } else {
      UNIMPLEMENTED();
    }
  }
  return op_confs;
}

void AddGlobalInputOutputCriticalSection(const HashSet<const OpNode*>& op_nodes,
                                         const std::vector<std::string>& lbi_producer_op_names,
                                         JobBuilder* job_builder) {
  auto* critical_section =
      Global<CriticalSectionDesc>::Get()->AddCriticalSection(GlobalJobDesc().job_id());
  {
    auto* io_cs = critical_section->mutable_input_output_critical_section();
    *io_cs->mutable_lbi_producer_op_name() = {lbi_producer_op_names.begin(),
                                              lbi_producer_op_names.end()};
  }
  auto time_shape = std::make_unique<Shape>(
      DimVector{GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()});
  HashMap<ParallelDesc, HashSet<const OpNode*>> parallel_desc2op_nodes;
  for (const OpNode* op_node : op_nodes) {
    CHECK(parallel_desc2op_nodes[op_node->parallel_desc()].insert(op_node).second);
  }
  std::vector<OperatorConf> source_ticks;
  std::vector<OperatorConf> sink_ticks;
  for (const auto& pair : parallel_desc2op_nodes) {
    source_ticks.push_back(PrependTick(pair.second, job_builder));
    for (const auto& sink_tick : AddTickForTimeShape(*time_shape, pair.second, job_builder)) {
      sink_ticks.push_back(sink_tick);
    }
  }
  OperatorConf src_subset_tick_op;
  {
    CHECK_EQ(source_ticks.empty(), false);
    CHECK_JUST(BuildSrcSubsetTickOpAndParallelConf(&src_subset_tick_op, job_builder));
    CHECK_JUST(
        CreateSourceTicksAndSrcSubsetTick(critical_section, &src_subset_tick_op, job_builder));
    for (auto& op_conf : source_ticks) {
      op_conf.mutable_tick_conf()->add_tick(src_subset_tick_op.name() + "/"
                                            + src_subset_tick_op.src_subset_tick_conf().out());
    }
    job_builder->MutOpsOnlyOnce(source_ticks);
  }
  HashSet<LogicalBlobId> tick_lbis;
  for (const auto& op_conf : sink_ticks) {
    LogicalBlobId lbi;
    lbi.set_op_name(op_conf.name());
    CHECK(op_conf.has_device_tick_conf());
    lbi.set_blob_name(op_conf.device_tick_conf().out());
    CHECK(tick_lbis.insert(lbi).second);
  }
  CHECK_JUST(CreateDstSubsetTickAndSinkTicks(critical_section, src_subset_tick_op, tick_lbis,
                                             job_builder));
}

}  // namespace

void AutoPrependTick(const OpGraph& op_graph, JobBuilder* job_builder) {
  PrependTickByParallelDesc(op_graph, job_builder);
  OperatorConf src_subset_tick_op;
  CHECK_JUST(BuildSrcSubsetTickOpAndParallelConf(&src_subset_tick_op, job_builder));
  CHECK_JUST(ConnectSrcSubsetTickAndOtherTick(src_subset_tick_op, job_builder));
}

void AddTickForTimeShape(const OpGraph& op_graph, JobBuilder* job_builder) {
  const auto& src_time_shape = *GetSrcSubsetTickOpNode(op_graph)->out_blob_time_shape();
  HashSet<const OpNode*> sink_op_nodes;
  op_graph.ForEachNode([&](OpNode* op_node) {
    CHECK(!op_node->op().op_conf().has_sink_tick_conf());
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt == 0) { sink_op_nodes.insert(op_node); }
  });
  AddTickForTimeShape(src_time_shape, sink_op_nodes, job_builder);
}

void AutoSourceAndSinkTick(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto* critical_section =
      Global<CriticalSectionDesc>::Get()->AddCriticalSection(GlobalJobDesc().job_id());
  critical_section->mutable_total_job_critical_section();
  op_graph.ForEachNode([&](OpNode* node) { CHECK(!node->op().op_conf().has_sink_tick_conf()); });
  const auto& src_time_shape = *GetSrcSubsetTickOpNode(op_graph)->out_blob_time_shape();
  HashSet<LogicalBlobId> tick_lbis;
  op_graph.ForEachNode([&](OpNode* op_node) {
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt > 0) { return; }
    CHECK(op_node->op().op_conf().has_device_tick_conf());
    CHECK(*op_node->out_blob_time_shape() == src_time_shape);
    CHECK(tick_lbis.emplace(op_node->op().BnInOp2Lbi(op_node->op().SoleObn())).second);
  });
  OperatorConf src_subset_tick = CHECK_JUST(FindSrcSubsetTickOpConf(job_builder->job()));
  CHECK_JUST(CreateSourceTicksAndSrcSubsetTick(critical_section, &src_subset_tick, job_builder));
  CHECK_JUST(
      CreateDstSubsetTickAndSinkTicks(critical_section, src_subset_tick, tick_lbis, job_builder));
}

void AddGlobalInputCriticalSections(const OpGraph& op_graph, JobBuilder* job_builder) {
  ForEachInputCriticalSectionOpNodes(
      op_graph, [&](const HashSet<const OpNode*>& op_nodes,
                    const std::vector<std::string>& lbi_producer_op_names) {
        AddGlobalInputOutputCriticalSection(op_nodes, lbi_producer_op_names, job_builder);
      });
}

void AddGlobalOutputCriticalSections(const OpGraph& op_graph, JobBuilder* job_builder) {
  ForEachOutputCriticalSectionOpNodes(
      op_graph, [&](const HashSet<const OpNode*>& op_nodes,
                    const std::vector<std::string>& lbi_producer_op_names) {
        AddGlobalInputOutputCriticalSection(op_nodes, lbi_producer_op_names, job_builder);
      });
}

}  // namespace oneflow
