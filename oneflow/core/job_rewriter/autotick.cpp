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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job_rewriter/autotick.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"

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
    parallel_desc2op_node[op_node->parallel_desc()].emplace_back(op_node);
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

Maybe<const OperatorConf&> FindJobSoleSrcSubsetTickOpConf(const Job& job) {
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
  for (int64_t machine_id : Singleton<ResourceDesc, ForSession>::Get()->process_ranks()) {
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
  }
  JUST(job_builder->AddOp(parallel_conf, *dst_subset_tick_op));
  return Maybe<void>::Ok();
}

Maybe<void> CreateDstSubsetTickAndSinkTicks(
    const OperatorConf& src_subset_tick, const HashSet<LogicalBlobId>& tick_lbis,
    JobBuilder* job_builder,
    const std::function<Maybe<void>(int64_t machine_id, const std::string& op_name)>& DoEachSink) {
  OperatorConf dst_subset_tick;
  dst_subset_tick.mutable_dst_subset_tick_conf()->add_in(
      src_subset_tick.name() + "/" + src_subset_tick.src_subset_tick_conf().out());
  JUST(BuildDstSubsetTickOpAndParallelConf(tick_lbis, &dst_subset_tick, job_builder));
  const auto& process_ranks = Singleton<ResourceDesc, ForSession>::Get()->process_ranks();
  HashMap<int64_t, std::string> machine_id2gather_tick_in_lbns;
  for (int64_t machine_id : process_ranks) {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
    OperatorConf tick_op;
    {
      tick_op.set_name("System-AutoTick-Tick_" + NewUniqueId());
      auto* tick_conf = tick_op.mutable_tick_conf();
      tick_conf->add_tick(dst_subset_tick.name() + "/"
                          + dst_subset_tick.dst_subset_tick_conf().out());
      tick_conf->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, tick_op));
    }
    CHECK_OR_RETURN(
        machine_id2gather_tick_in_lbns.emplace(machine_id, tick_op.name() + "/out").second);
  }
  for (int64_t machine_id : process_ranks) {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
    OperatorConf tick_op;
    {
      tick_op.set_name("System-SyncAllRanksSinkTick_" + NewUniqueId());
      auto* tick_conf = tick_op.mutable_tick_conf();
      // gather ticks from all processes.
      for (int64_t tick_machine_id : process_ranks) {
        tick_conf->add_tick(JUST(MapAt(machine_id2gather_tick_in_lbns, tick_machine_id)));
      }
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
    JUST(DoEachSink(machine_id, sink_tick_op.name()));
  }
  return Maybe<void>::Ok();
}

Maybe<void> CreateDstSubsetTickAndSinkTicks(CriticalSection* critical_section,
                                            const OperatorConf& src_subset_tick,
                                            const HashSet<LogicalBlobId>& tick_lbis,
                                            JobBuilder* job_builder) {
  auto* map = critical_section->mutable_machine_id2sink_tick_op_name();
  const auto& DoEachSink = [&](int64_t machine_id, const std::string& op_name) -> Maybe<void> {
    (*map)[machine_id] = op_name;
    return Maybe<void>::Ok();
  };
  JUST(CreateDstSubsetTickAndSinkTicks(src_subset_tick, tick_lbis, job_builder, DoEachSink));
  return Maybe<void>::Ok();
}

Maybe<void> BuildSrcSubsetTickOpAndParallelConf(OperatorConf* src_subset_tick_op,
                                                JobBuilder* job_builder) {
  src_subset_tick_op->set_name("System-AutoTick-SrcSubsetTick_" + NewUniqueId());
  src_subset_tick_op->mutable_src_subset_tick_conf()->set_out("out");
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  for (int64_t machine_id : Singleton<ResourceDesc, ForSession>::Get()->process_ranks()) {
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
  }
  JUST(job_builder->AddOp(parallel_conf, *src_subset_tick_op));
  return Maybe<void>::Ok();
}

Maybe<void> CreateSourceTicksAndSrcSubsetTick(
    OperatorConf* src_subset_tick_op, JobBuilder* job_builder,
    const std::function<Maybe<void>(int64_t machine_id, const std::string& op_name)>& DoEachSrc) {
  for (int64_t machine_id : Singleton<ResourceDesc, ForSession>::Get()->process_ranks()) {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
    OperatorConf src_tick_op;
    {
      src_tick_op.set_name("System-AutoTick-SourceTick_" + NewUniqueId());
      src_tick_op.mutable_source_tick_conf()->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, src_tick_op));
    }
    JUST(DoEachSrc(machine_id, src_tick_op.name()));
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

Maybe<void> CreateSourceTicksAndSrcSubsetTick(CriticalSection* critical_section,
                                              OperatorConf* src_subset_tick_op,
                                              JobBuilder* job_builder) {
  auto* map = critical_section->mutable_machine_id2source_tick_op_name();
  const auto& DoEachSrc = [&](int64_t machine_id, const std::string& op_name) -> Maybe<void> {
    (*map)[machine_id] = op_name;
    return Maybe<void>::Ok();
  };
  JUST(CreateSourceTicksAndSrcSubsetTick(src_subset_tick_op, job_builder, DoEachSrc));
  return Maybe<void>::Ok();
}

Maybe<void> ConnectSrcSubsetTickAndOtherTick(const OperatorConf& src_subset_tick_op,
                                             JobBuilder* job_builder) {
  CHECK_OR_RETURN(src_subset_tick_op.has_src_subset_tick_conf());
  const std::string& src_lbn =
      src_subset_tick_op.name() + "/" + src_subset_tick_op.src_subset_tick_conf().out();
  JUST(job_builder->ForEachOperator([&](const Operator& op) -> Maybe<void> {
    if (op.op_name() != src_subset_tick_op.name()) {
      CHECK_OR_RETURN(!op.op_conf().has_src_subset_tick_conf());
    }
    auto mut_helper = NewMutOpConTickInputHelper(op.op_conf());
    if (!mut_helper) { return Maybe<void>::Ok(); }
    if (mut_helper->IsTickInputBound() == true) { return Maybe<void>::Ok(); }
    JUST(job_builder->MutOpOnlyOnce(mut_helper->NewTickInputBoundOpConf(src_lbn)));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<const OpNode*> GetSrcSubsetTickOpNode(const OpGraph& op_graph) {
  const OpNode* src_subset_tick = nullptr;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    if (op_node->op().op_conf().has_src_subset_tick_conf()) {
      CHECK_ISNULL_OR_RETURN(src_subset_tick);
      src_subset_tick = op_node;
    }
    return Maybe<void>::Ok();
  }));
  CHECK_NOTNULL_OR_RETURN(src_subset_tick);
  return src_subset_tick;
}

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
                        const std::shared_ptr<const Shape>& time_shape, ParallelConf parallel_conf,
                        JobBuilder* job_builder) {
  OperatorConf device_tick_op_conf = MakeDeviceTickOpConf(tick_name);
  if (time_shape) {
    time_shape->ToProto(device_tick_op_conf.mutable_device_tick_conf()->mutable_time_shape());
  }
  for (const auto& op_name : op_names) { device_tick_op_conf.add_ctrl_in_op_name(op_name); }
  job_builder->AddOps(parallel_conf, {device_tick_op_conf});
  return device_tick_op_conf;
}

OperatorConf AppendTick(const std::string tick_name, const std::list<const OpNode*>& op_nodes,
                        const std::shared_ptr<const Shape>& time_shape, JobBuilder* job_builder) {
  std::vector<std::string> op_names;
  op_names.reserve(op_nodes.size());
  for (const auto* op_node : op_nodes) {
    CHECK(op_nodes.front()->parallel_desc() == op_node->parallel_desc());
    op_names.emplace_back(op_node->op().op_name());
  }
  return AppendTick(tick_name, op_names, time_shape,
                    op_nodes.front()->parallel_desc().parallel_conf(), job_builder);
}

OperatorConf PrependTick(const HashSet<const OpNode*>& op_nodes, JobBuilder* job_builder) {
  CHECK_GE(op_nodes.size(), 1);
  OperatorConf tick_op_conf = MakeTickOpConf("Prepend");
  std::vector<OperatorConf> op_confs;
  op_confs.reserve(op_nodes.size());
  for (const OpNode* op_node : op_nodes) {
    OperatorConf op_conf(op_node->op().op_conf());
    op_conf.add_ctrl_in_op_name(tick_op_conf.name());
    op_confs.emplace_back(op_conf);
  }
  job_builder->MutOpsOnlyOnce({op_confs});
  ParallelDesc pd((*op_nodes.begin())->parallel_desc());
  pd.set_device_type(DeviceType::kCPU);
  job_builder->AddOps(pd.parallel_conf(), {tick_op_conf});
  return tick_op_conf;
}

OperatorConf AppendAccTick(const Shape& src_shape, const std::list<const OpNode*>& op_nodes,
                           JobBuilder* job_builder) {
  std::shared_ptr<const Shape> tick_shape = CHECK_JUST(op_nodes.front()->op().GetOpTimeShape());
  CHECK_EQ(tick_shape->elem_cnt() % src_shape.elem_cnt(), 0);
  const OperatorConf& tick_op_conf = AppendTick("AppendAcc", op_nodes, tick_shape, job_builder);
  OperatorConf acc_op_conf;
  {
    acc_op_conf.set_name(std::string("System-AutoTick-AccTick_") + NewUniqueId());
    auto* acc_conf = acc_op_conf.mutable_acc_tick_conf();
    CHECK(tick_op_conf.has_device_tick_conf());
    acc_conf->set_one(tick_op_conf.name() + "/" + tick_op_conf.device_tick_conf().out());
    acc_conf->set_acc("acc");
    acc_conf->set_max_acc_num(tick_shape->elem_cnt() / src_shape.elem_cnt());
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
  ret.reserve(op_nodes.size());
  for (const OpNode* op_node : op_nodes) { ret.emplace_back(op_node->op().op_name()); }
  return ret;
};

Maybe<void> InitOpTypeCase2OpNodes(
    const OpGraph& op_graph,
    HashMap<OperatorConf::OpTypeCase, HashSet<const OpNode*>>* op_type_case2op_nodes) {
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    const auto& op_conf = op_node->op().op_conf();
    if (IsInterfaceOpConf(op_conf)) {
      CHECK_OR_RETURN((*op_type_case2op_nodes)[op_conf.op_type_case()].emplace(op_node).second);
    }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> ForEachInputCriticalSectionOpNodes(
    const OpGraph& op_graph,
    const std::function<Maybe<void>(const HashSet<const OpNode*>&,
                                    const std::vector<std::string>&)>& Handler) {
  HashMap<OperatorConf::OpTypeCase, HashSet<const OpNode*>> op_type_case2op_nodes;
  JUST(InitOpTypeCase2OpNodes(op_graph, &op_type_case2op_nodes));
  OperatorConf::OpTypeCase op_type_case = OperatorConf::kInputConf;
  if (op_type_case2op_nodes[op_type_case].empty()) { return Maybe<void>::Ok(); }
  HashSet<const OpNode*> op_nodes = op_type_case2op_nodes[op_type_case];
  for (const OpNode* op_node : op_type_case2op_nodes[op_type_case]) {
    op_node->ForEachNodeOnOutEdge([&](OpNode* out_node) { op_nodes.insert(out_node); });
  }
  JUST(Handler(op_nodes, GetOpNames(op_type_case2op_nodes[op_type_case])));
  return Maybe<void>::Ok();
}

Maybe<void> ForEachOutputCriticalSectionOpNodes(
    const OpGraph& op_graph,
    const std::function<Maybe<void>(const HashSet<const OpNode*>&,
                                    const std::vector<std::string>&)>& Handler) {
  HashMap<OperatorConf::OpTypeCase, HashSet<const OpNode*>> op_type_case2op_nodes;
  JUST(InitOpTypeCase2OpNodes(op_graph, &op_type_case2op_nodes));
  if (op_type_case2op_nodes[OperatorConf::kReturnConf].empty() == false) {
    JUST(Handler(op_type_case2op_nodes[OperatorConf::kReturnConf],
                 GetOpNames(op_type_case2op_nodes[OperatorConf::kReturnConf])));
  }
  if (op_type_case2op_nodes[OperatorConf::kOutputConf].empty() == false) {
    JUST(Handler(op_type_case2op_nodes[OperatorConf::kOutputConf],
                 GetOpNames(op_type_case2op_nodes[OperatorConf::kOutputConf])));
  }
  return Maybe<void>::Ok();
}

Maybe<std::vector<OperatorConf>> AddTickForTimeShape(const Shape& src_time_shape,
                                                     const HashSet<const OpNode*>& op_nodes,
                                                     JobBuilder* job_builder) {
  HashMap<std::pair<ParallelDesc, std::pair<Shape, Shape>>, std::list<const OpNode*>>
      pd7ts2op_nodes;
  for (const OpNode* op_node : op_nodes) {
    auto ts = std::make_pair(*JUST(op_node->op().GetInputOutputFastestTimeShape()),
                             *JUST(op_node->op().GetOpTimeShape()));
    pd7ts2op_nodes[{op_node->parallel_desc(), ts}].emplace_back(op_node);
  }
  std::vector<OperatorConf> op_confs;
  op_confs.reserve(pd7ts2op_nodes.size());
  for (const auto& pair : pd7ts2op_nodes) {
    const std::pair<Shape, Shape>& ts = pair.first.second;
    if (ts.second.elem_cnt() == src_time_shape.elem_cnt()) {
      CHECK_GE_OR_RETURN(ts.first.elem_cnt(), ts.second.elem_cnt());
      op_confs.emplace_back(
          AppendTick("Append", pair.second, std::make_shared<const Shape>(ts.second), job_builder));
    } else if (ts.second.elem_cnt() > src_time_shape.elem_cnt()) {
      op_confs.emplace_back(AppendAccTick(src_time_shape, pair.second, job_builder));
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  return op_confs;
}

Maybe<void> AddGlobalInputOutputCriticalSection(
    const HashSet<const OpNode*>& op_nodes, const std::vector<std::string>& lbi_producer_op_names,
    JobBuilder* job_builder) {
  auto* critical_section =
      Singleton<CriticalSectionDesc>::Get()->AddCriticalSection(GlobalJobDesc().job_id());
  {
    auto* io_cs = critical_section->mutable_input_output_critical_section();
    *io_cs->mutable_lbi_producer_op_name() = {lbi_producer_op_names.begin(),
                                              lbi_producer_op_names.end()};
  }
  auto time_shape = std::make_unique<Shape>(DimVector{1, 1});
  HashMap<ParallelDesc, HashSet<const OpNode*>> parallel_desc2op_nodes;
  for (const OpNode* op_node : op_nodes) {
    CHECK_OR_RETURN(parallel_desc2op_nodes[op_node->parallel_desc()].insert(op_node).second);
  }
  std::vector<OperatorConf> source_ticks;
  std::vector<OperatorConf> sink_ticks;
  source_ticks.reserve(parallel_desc2op_nodes.size());
  for (const auto& pair : parallel_desc2op_nodes) {
    source_ticks.emplace_back(PrependTick(pair.second, job_builder));
    const auto& ops = JUST(AddTickForTimeShape(*time_shape, pair.second, job_builder));
    for (const auto& sink_tick : *ops) { sink_ticks.emplace_back(sink_tick); }
  }
  OperatorConf src_subset_tick_op;
  {
    CHECK_EQ_OR_RETURN(source_ticks.empty(), false);
    JUST(BuildSrcSubsetTickOpAndParallelConf(&src_subset_tick_op, job_builder));
    JUST(CreateSourceTicksAndSrcSubsetTick(critical_section, &src_subset_tick_op, job_builder));
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
    CHECK_OR_RETURN(op_conf.has_device_tick_conf());
    lbi.set_blob_name(op_conf.device_tick_conf().out());
    CHECK_OR_RETURN(tick_lbis.insert(lbi).second);
  }
  JUST(CreateDstSubsetTickAndSinkTicks(critical_section, src_subset_tick_op, tick_lbis,
                                       job_builder));
  return Maybe<void>::Ok();
}

Maybe<void> MultiClientAddOneWaitAndSendIdsOp(JobBuilder* job_builder, int64_t machine_id,
                                              const OperatorConf& src_op_consumer) {
  ParallelConf parallel_conf;
  {
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
  }

  // add wait_and_send_ids op conf
  OperatorConf wait_and_send_ids_op_conf;
  {
    wait_and_send_ids_op_conf.set_name(std::string("System-Src-WaitAndSendIds_") + NewUniqueId());
    wait_and_send_ids_op_conf.set_pass_tag(kMainOp);
    auto* wait_and_send_ids_conf = wait_and_send_ids_op_conf.mutable_wait_and_send_ids_conf();
    wait_and_send_ids_conf->set_out("out");
    wait_and_send_ids_conf->set_wait_buffer_name("UnimplementedBufferName");
    wait_and_send_ids_conf->set_data_type(DataType::kInt32);
    // wait_and_send_ids_conf->id_list() is unused in multi-client mode.
  }
  JUST(job_builder->AddOp(parallel_conf, wait_and_send_ids_op_conf));

  // connect wait_and_send_ids to tick op which was connected to the src tick op
  OperatorConf tick_op_conf;
  tick_op_conf.CopyFrom(src_op_consumer);
  CHECK_OR_RETURN(tick_op_conf.has_tick_conf());
  CHECK_EQ_OR_RETURN(tick_op_conf.tick_conf().tick_size(), 1);
  tick_op_conf.mutable_tick_conf()->clear_tick();
  tick_op_conf.mutable_tick_conf()->add_tick(
      GenLogicalBlobName(wait_and_send_ids_op_conf.name(), "out"));
  JUST(job_builder->MutOpOnlyOnce(tick_op_conf));

  return Maybe<void>::Ok();
}

Maybe<void> MultiClientAddWaitAndSendIds(
    JobBuilder* job_builder, const HashMap<int64_t, std::string>& machine_id2src_op_name) {
  // Prepare the consumer tick op for each Source op
  HashMap<std::string, OperatorConf> src_op_name2solo_consumer_tick_op;
  HashSet<std::string> src_op_names;
  for (const auto& pair : machine_id2src_op_name) {
    CHECK_OR_RETURN(src_op_names.insert(pair.second).second)
        << " duplicated src op name " << pair.second;
  }
  JUST(job_builder->ForEachOperator([&](const Operator& op) -> Maybe<void> {
    // skip if the op is not a tick op
    if (!op.op_conf().has_tick_conf()) { return Maybe<void>::Ok(); }
    for (const auto& ibn : op.input_bns()) {
      const auto& input_lbi = op.BnInOp2Lbi(ibn);
      if (src_op_names.count(input_lbi.op_name()) == 0) { continue; }
      auto insert_pair =
          src_op_name2solo_consumer_tick_op.emplace(input_lbi.op_name(), op.op_conf());
      CHECK_OR_RETURN(insert_pair.second)
          << " Duplicated src op name " << input_lbi.op_name() << " old op "
          << insert_pair.first->second.DebugString() << " new op " << op.op_conf().DebugString();
    }
    return Maybe<void>::Ok();
  }));

  // Replace Source op with WaitAndSendIds op
  for (const auto& pair : machine_id2src_op_name) {
    auto tick_op_iter = src_op_name2solo_consumer_tick_op.find(pair.second);
    CHECK_OR_RETURN(tick_op_iter != src_op_name2solo_consumer_tick_op.end())
        << "Can't find consumer tick op of source op name " << pair.second << " machine id "
        << pair.first;
    JUST(MultiClientAddOneWaitAndSendIdsOp(job_builder, pair.first, tick_op_iter->second));
  }

  // Delete Source op
  std::vector<std::string> src_op_name_vec{src_op_names.begin(), src_op_names.end()};
  job_builder->DelOps(src_op_name_vec);

  return Maybe<void>::Ok();
}

Maybe<void> MultiClientAddCallbackNotifier(JobBuilder* job_builder, int64_t machine_id,
                                           const std::string& sink_op_name) {
  ParallelConf parallel_conf;
  {
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":0");
  }
  OperatorConf callback_notify_op_conf;
  {
    callback_notify_op_conf.set_name(std::string("System-Sink-CallbackNotify_") + NewUniqueId());
    callback_notify_op_conf.set_pass_tag(kMainOp);
    auto* callback_notify_conf = callback_notify_op_conf.mutable_callback_notify_conf();
    callback_notify_conf->set_in(GenLogicalBlobName(sink_op_name, "out"));
    // callback_notify_conf->callback_buffer_name() is unused in multi-client mode.
  }
  JUST(job_builder->AddOp(parallel_conf, callback_notify_op_conf));
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> AutoPrependTick(const OpGraph& op_graph, JobBuilder* job_builder) {
  PrependTickByParallelDesc(op_graph, job_builder);
  OperatorConf src_subset_tick_op;
  JUST(BuildSrcSubsetTickOpAndParallelConf(&src_subset_tick_op, job_builder));
  JUST(ConnectSrcSubsetTickAndOtherTick(src_subset_tick_op, job_builder));
  return Maybe<void>::Ok();
}

Maybe<void> AddTickForTimeShape(const OpGraph& op_graph, JobBuilder* job_builder) {
  const auto* op_node = JUST(GetSrcSubsetTickOpNode(op_graph));
  const auto& src_time_shape = *JUST(op_node->op().GetOpTimeShape());
  HashSet<const OpNode*> sink_op_nodes;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    CHECK_OR_RETURN(!op_node->op().op_conf().has_sink_tick_conf());
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt == 0) { sink_op_nodes.insert(op_node); }
    return Maybe<void>::Ok();
  }));
  JUST(AddTickForTimeShape(src_time_shape, sink_op_nodes, job_builder));
  return Maybe<void>::Ok();
}

Maybe<void> AutoSourceAndSinkTick(
    const OpGraph& op_graph, JobBuilder* job_builder,
    const std::function<Maybe<void>(int64_t machine_id, const std::string& op_name)>& DoEachSrc,
    const std::function<Maybe<void>(int64_t machine_id, const std::string& op_name)>& DoEachSink) {
  JUST(op_graph.MaybeForEachNode([&](OpNode* node) -> Maybe<void> {
    CHECK_OR_RETURN(!node->op().op_conf().has_sink_tick_conf());
    return Maybe<void>::Ok();
  }));
  const auto* op_node = JUST(GetSrcSubsetTickOpNode(op_graph));
  const auto& src_time_shape = JUST(op_node->op().GetOpTimeShape());
  HashSet<LogicalBlobId> tick_lbis;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt > 0) { return Maybe<void>::Ok(); }
    CHECK_OR_RETURN(op_node->op().op_conf().has_device_tick_conf());
    CHECK_OR_RETURN(JUST(op_node->op().GetOpTimeShape())->elem_cnt() == src_time_shape->elem_cnt());
    CHECK_OR_RETURN(tick_lbis.emplace(op_node->op().BnInOp2Lbi(op_node->op().SoleObn())).second);
    return Maybe<void>::Ok();
  }));
  OperatorConf src_subset_tick = JUST(FindJobSoleSrcSubsetTickOpConf(job_builder->job()));
  JUST(CreateSourceTicksAndSrcSubsetTick(&src_subset_tick, job_builder, DoEachSrc));
  JUST(CreateDstSubsetTickAndSinkTicks(src_subset_tick, tick_lbis, job_builder, DoEachSink));
  return Maybe<void>::Ok();
}

Maybe<void> MultiClientAutoSourceAndSinkTick(const OpGraph& op_graph, Job* job) {
  HashMap<int64_t, std::string> machine_id2src_op_name;
  HashMap<int64_t, std::string> machine_id2sink_op_name;
  {
    JobBuilder job_builder(job);
    const auto& DoEachSrc = [&](int64_t machine_id, const std::string& op_name) -> Maybe<void> {
      CHECK_OR_RETURN(machine_id2src_op_name.emplace(machine_id, op_name).second);
      return Maybe<void>::Ok();
    };
    const auto& DoEachSink = [&](int64_t machine_id, const std::string& op_name) -> Maybe<void> {
      CHECK_OR_RETURN(machine_id2sink_op_name.emplace(machine_id, op_name).second);
      return Maybe<void>::Ok();
    };
    JUST(AutoSourceAndSinkTick(op_graph, &job_builder, DoEachSrc, DoEachSink));
  }
  {
    JobBuilder job_builder(job);
    JUST(MultiClientAddWaitAndSendIds(&job_builder, machine_id2src_op_name));

    for (const auto& pair : machine_id2sink_op_name) {
      JUST(MultiClientAddCallbackNotifier(&job_builder, pair.first, pair.second));
    }
  }
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> InsertCriticalSectionSrcAndDstTicks(
    const std::vector<const OpNode*>& interface_op_nodes, JobBuilder* job_builder,
    std::vector<std::string>* interface_src_tick_op_names,
    std::vector<std::string>* interface_dst_tick_lbns) {
  HashMap<ParallelDesc, std::vector<const OpNode*>> parallel_desc2interface_op_nodes;
  for (const auto* op_node : interface_op_nodes) {
    parallel_desc2interface_op_nodes[op_node->parallel_desc()].push_back(op_node);
  }
  for (const auto& pair : parallel_desc2interface_op_nodes) {
    const auto& parallel_conf = pair.first.parallel_conf();
    for (const auto* op_node : pair.second) {
      OperatorConf interface_op(op_node->op().op_conf());
      {
        OperatorConf device_tick_op;
        device_tick_op.set_name("System-EagerCriticalSection-Interface-Begin-Tick-"
                                + NewUniqueId());
        auto* device_tick_op_conf = device_tick_op.mutable_device_tick_conf();
        device_tick_op_conf->set_out("out");
        interface_src_tick_op_names->push_back(device_tick_op.name());
        JUST(job_builder->AddOp(parallel_conf, device_tick_op));
        interface_op.add_ctrl_in_op_name(device_tick_op.name());
        JUST(job_builder->MutOpOnlyOnce(interface_op));
      }
      {
        OperatorConf device_tick_op;
        device_tick_op.set_name("System-EagerCriticalSection-Interface-End-Tick-" + NewUniqueId());
        device_tick_op.add_ctrl_in_op_name(interface_op.name());
        auto* device_tick_op_conf = device_tick_op.mutable_device_tick_conf();
        device_tick_op_conf->set_out("out");
        interface_dst_tick_lbns->push_back(device_tick_op.name() + "/out");
        JUST(job_builder->AddOp(parallel_conf, device_tick_op));
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InsertSrcSubsetTickAndDstSubsetTick(
    const std::vector<std::string>& interface_src_tick_op_names,
    const std::vector<std::string>& interface_dst_tick_lbns, JobBuilder* job_builder,
    std::string* src_subset_tick_op_name, LogicalBlobId* dst_subset_tick_lbi) {
  {
    OperatorConf src_subset_tick;
    JUST(BuildSrcSubsetTickOpAndParallelConf(&src_subset_tick, job_builder));
    *src_subset_tick_op_name = src_subset_tick.name();
  }
  for (const auto& op_name : interface_src_tick_op_names) {
    OperatorConf op_conf(JUST(job_builder->OpConf4OpName(op_name)));
    CHECK_OR_RETURN(op_conf.has_device_tick_conf());
    op_conf.mutable_device_tick_conf()->add_tick(*src_subset_tick_op_name + "/out");
    JUST(job_builder->MutOpOnlyOnce(op_conf));
  }
  HashSet<LogicalBlobId> dst_subset_tick_input_lbis;
  dst_subset_tick_input_lbis.insert(GenLogicalBlobId(*src_subset_tick_op_name + "/out"));
  for (const auto& lbn : interface_dst_tick_lbns) {
    const auto& lbi = GenLogicalBlobId(lbn);
    CHECK_OR_RETURN(dst_subset_tick_input_lbis.insert(lbi).second);
  }
  {
    OperatorConf dst_subset_tick_op;
    JUST(BuildDstSubsetTickOpAndParallelConf(dst_subset_tick_input_lbis, &dst_subset_tick_op,
                                             job_builder));
    dst_subset_tick_lbi->set_op_name(dst_subset_tick_op.name());
    CHECK_OR_RETURN(dst_subset_tick_op.has_dst_subset_tick_conf());
    dst_subset_tick_lbi->set_blob_name(dst_subset_tick_op.dst_subset_tick_conf().out());
  }
  return Maybe<void>::Ok();
}

Maybe<void> InsertCriticalSectionWaitTicks(const OpGraph& op_graph, JobBuilder* job_builder,
                                           const std::string& src_subset_tick_op_name,
                                           const std::string& wait_buffer_name) {
  std::vector<const OpNode*> wait_and_send_id_op_nodes;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (!op_node->op().op_conf().has_wait_and_send_ids_conf()) { return; }
    wait_and_send_id_op_nodes.push_back(op_node);
  });
  CHECK_GT_OR_RETURN(wait_and_send_id_op_nodes.size(), 0);
  OperatorConf src_subset_tick_op(JUST(job_builder->OpConf4OpName(src_subset_tick_op_name)));
  CHECK_OR_RETURN(src_subset_tick_op.has_src_subset_tick_conf());
  for (const OpNode* wait_and_send_id_op_node : wait_and_send_id_op_nodes) {
    LogicalBlobId lbi;
    lbi.set_op_name(wait_and_send_id_op_node->op().op_name());
    lbi.set_blob_name(wait_and_send_id_op_node->op().op_conf().wait_and_send_ids_conf().out());
    OperatorConf critical_section_wait_op;
    {
      critical_section_wait_op.set_name("System-EagerCriticalSection-Wait-" + NewUniqueId());
      auto* conf = critical_section_wait_op.mutable_critical_section_wait_tick_conf();
      conf->add_tick(GenLogicalBlobName(lbi));
      conf->set_out("out");
      conf->set_buffer_name(wait_buffer_name);
    }
    const auto& parallel_conf = wait_and_send_id_op_node->parallel_desc().parallel_conf();
    JUST(job_builder->AddOp(parallel_conf, critical_section_wait_op));
    src_subset_tick_op.mutable_src_subset_tick_conf()->add_in(critical_section_wait_op.name()
                                                              + "/out");
  }
  JUST(job_builder->MutOpOnlyOnce(src_subset_tick_op));
  return Maybe<void>::Ok();
}

Maybe<LogicalBlobId> InsertCriticalSectionCallbackTicks(const OpGraph& op_graph,
                                                        JobBuilder* job_builder,
                                                        const LogicalBlobId& dst_subset_tick_lbi,
                                                        const std::string& callback_buffer_name) {
  OperatorConf critical_section_callback_op;
  critical_section_callback_op.set_name("System-EagerCriticalSection-Callback-" + NewUniqueId());
  auto* conf = critical_section_callback_op.mutable_critical_section_callback_tick_conf();
  conf->add_tick(GenLogicalBlobName(dst_subset_tick_lbi));
  conf->set_out("out");
  conf->set_buffer_name(callback_buffer_name);
  const auto& op_name = dst_subset_tick_lbi.op_name();
  const auto& parallel_conf = JUST(job_builder->ParallelConf4OpName(op_name));
  JUST(job_builder->AddOp(parallel_conf, critical_section_callback_op));
  LogicalBlobId critical_section_callback_lbi;
  critical_section_callback_lbi.set_op_name(critical_section_callback_op.name());
  critical_section_callback_lbi.set_blob_name("out");
  return critical_section_callback_lbi;
}

Maybe<LogicalBlobId> MultiClientAutoCriticalSectionTick(
    const OpGraph& op_graph, JobBuilder* job_builder,
    const std::vector<const OpNode*>& interface_op_nodes, const std::string& wait_buffer_name,
    const std::string& callback_buffer_name) {
  std::vector<std::string> interface_src_tick_op_names;
  std::vector<std::string> interface_dst_tick_lbns;
  JUST(InsertCriticalSectionSrcAndDstTicks(interface_op_nodes, job_builder,
                                           &interface_src_tick_op_names, &interface_dst_tick_lbns));
  std::string src_subset_tick_op_name;
  LogicalBlobId dst_subset_tick_lbi;
  JUST(InsertSrcSubsetTickAndDstSubsetTick(interface_src_tick_op_names, interface_dst_tick_lbns,
                                           job_builder, &src_subset_tick_op_name,
                                           &dst_subset_tick_lbi));
  JUST(InsertCriticalSectionWaitTicks(op_graph, job_builder, src_subset_tick_op_name,
                                      wait_buffer_name));
  const auto& lbi = JUST(InsertCriticalSectionCallbackTicks(
      op_graph, job_builder, dst_subset_tick_lbi, callback_buffer_name));
  return lbi;
}

Maybe<void> ConnectCriticalSectionCallbackToJobSoleDstSubsetTick(
    const OpGraph& op_graph, JobBuilder* job_builder,
    const std::vector<std::shared_ptr<LogicalBlobId>>& critical_section_callback_lbis) {
  const OpNode* dst_subset_tick_op_node = nullptr;
  JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
    if (!op_node->op().op_conf().has_dst_subset_tick_conf()) { return Maybe<void>::Ok(); }
    CHECK_OR_RETURN(dst_subset_tick_op_node == nullptr);
    dst_subset_tick_op_node = op_node;
    return Maybe<void>::Ok();
  }));
  CHECK_NOTNULL_OR_RETURN(dst_subset_tick_op_node);
  OperatorConf dst_subset_tick_op(dst_subset_tick_op_node->op().op_conf());
  auto* conf = dst_subset_tick_op.mutable_dst_subset_tick_conf();
  for (const auto& lbi : critical_section_callback_lbis) { conf->add_in(GenLogicalBlobName(*lbi)); }
  JUST(job_builder->MutOpOnlyOnce(dst_subset_tick_op));
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> MultiClientAutoInterfaceCriticalSectionTick(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  std::vector<std::shared_ptr<LogicalBlobId>> critical_section_callback_lbis;
  {
    std::vector<const OpNode*> interface_op_nodes;
    op_graph.ForEachNode([&](OpNode* node) {
      if (node->op().op_conf().has_input_conf()) { interface_op_nodes.push_back(node); }
    });
    const auto& lbi = JUST(MultiClientAutoCriticalSectionTick(
        op_graph, &job_builder, interface_op_nodes,
        GetInputCriticalSectionWaitBufferName(job->job_conf().job_name()),
        GetInputCriticalSectionCallbackBufferName(job->job_conf().job_name())));
    critical_section_callback_lbis.push_back(lbi);
  }
  {
    std::vector<const OpNode*> interface_op_nodes;
    op_graph.ForEachNode([&](OpNode* node) {
      if (node->op().op_conf().has_output_conf()) { interface_op_nodes.push_back(node); }
    });
    const auto& lbi = JUST(MultiClientAutoCriticalSectionTick(
        op_graph, &job_builder, interface_op_nodes,
        GetOutputCriticalSectionWaitBufferName(job->job_conf().job_name()),
        GetOutputCriticalSectionCallbackBufferName(job->job_conf().job_name())));
    critical_section_callback_lbis.push_back(lbi);
  }
  JUST(ConnectCriticalSectionCallbackToJobSoleDstSubsetTick(op_graph, &job_builder,
                                                            critical_section_callback_lbis));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
