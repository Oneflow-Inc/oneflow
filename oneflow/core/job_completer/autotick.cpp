#include "oneflow/core/job_completer/autotick.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/critical_section_desc.h"

namespace oneflow {

namespace {

std::unique_ptr<MutOpConTickInputHelper> NewMutOpConTickInputHelper(const OperatorConf& op_conf) {
  std::unique_ptr<MutOpConTickInputHelper> ret;
  if (IsClassRegistered<MutOpConTickInputHelper>(op_conf.op_type_case())) {
    ret.reset(NewObj<MutOpConTickInputHelper>(op_conf.op_type_case()));
    ret->InitFromOpConf(op_conf);
  }
  return ret;
}

void GroupTickByParallelDesc(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  HashMap<ParallelDesc, std::vector<OpNode*>> parallel_desc2op_node;
  op_graph.ForEachNode([&](OpNode* op_node) {
    auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
    if (!mut_tick_input_helper) { return; }
    if (mut_tick_input_helper->IsTickInputBound() == true) { return; }
    parallel_desc2op_node[op_node->parallel_desc()].push_back(op_node);
  });
  for (const auto& pair : parallel_desc2op_node) {
    if (pair.second.size() == 1) { continue; }
    OperatorConf tick_op;
    tick_op.set_name("System-AutoTick-Tick_" + NewUniqueId());
    tick_op.mutable_tick_conf()->set_out("out");
    job_builder.AddOps(pair.first.parallel_conf(), {tick_op});

    for (const auto* op_node : pair.second) {
      auto mut_tick_input_helper = NewMutOpConTickInputHelper(op_node->op().op_conf());
      job_builder.MutOps({mut_tick_input_helper->NewTickInputBoundOpConf(tick_op.name() + "/out")});
    }
  }
}

void BuildSourceTickOpAndParallelConf(OperatorConf* src_tick_op, JobBuilder* job_builder) {
  src_tick_op->set_name("System-AutoTick-SourceTick_" + NewUniqueId());
  src_tick_op->mutable_source_tick_conf()->set_out("out");
  ParallelConf parallel_conf;
  parallel_conf.set_policy(kDataParallel);
  parallel_conf.add_device_name("0:cpu:0");
  job_builder->AddOps(parallel_conf, {*src_tick_op});
}

void BuildSinkTickOpAndParallelConf(OperatorConf* sink_tick_op, JobBuilder* job_builder) {
  sink_tick_op->set_name("System-AutoTick-SinkTick_" + NewUniqueId());
  sink_tick_op->mutable_sink_tick_conf();
  ParallelConf parallel_conf;
  parallel_conf.set_policy(kDataParallel);
  parallel_conf.add_device_name("0:cpu:0");
  job_builder->AddOps(parallel_conf, {*sink_tick_op});
}

void ConnectSourceTickAndOtherTick(Job* job) {
  JobBuilder job_builder(job);
  OperatorConf src_tick_op;
  BuildSourceTickOpAndParallelConf(&src_tick_op, &job_builder);

  job_builder.ForEachOperator([&](const Operator& op) {
    if (op.op_name() != src_tick_op.name()) { CHECK(!op.op_conf().has_source_tick_conf()); }
    auto mut_helper = NewMutOpConTickInputHelper(op.op_conf());
    if (!mut_helper) { return; }
    if (mut_helper->IsTickInputBound() == true) { return; }
    job_builder.MutOps({mut_helper->NewTickInputBoundOpConf(
        src_tick_op.name() + "/" + src_tick_op.source_tick_conf().out())});
  });
}

const OpNode* GetSrcTickOpNode(const OpGraph& op_graph) {
  const OpNode* src_tick = nullptr;
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_source_tick_conf()) {
      CHECK_ISNULL(src_tick);
      src_tick = op_node;
    }
  });
  CHECK_NOTNULL(src_tick);
  return src_tick;
}

const Shape& GetOpTimeShape(const OpNode* op_node) {
  const Shape* output_shape = op_node->out_blob_time_shape();
  if (output_shape == nullptr) { output_shape = op_node->GetInputBlobFastestTimeShape(); }
  return *output_shape;
};

OperatorConf MakeTickOpConf() {
  OperatorConf tick_op_conf;
  tick_op_conf.set_name(std::string("System-AutoTick-Tick_") + NewUniqueId());
  auto* tick_conf = tick_op_conf.mutable_tick_conf();
  tick_conf->set_out("out");
  return tick_op_conf;
}

OperatorConf AppendTick(const std::vector<std::string> op_names,
                        const std::vector<LogicalBlobId>& lbis, ParallelConf parallel_conf,
                        JobBuilder* job_builder) {
  OperatorConf tick_op_conf = MakeTickOpConf();
  for (const auto& op_name : op_names) { tick_op_conf.add_ctrl_in_op_name(op_name); }
  auto* tick_conf = tick_op_conf.mutable_tick_conf();
  for (const auto& lbi : lbis) { tick_conf->add_tick(GenLogicalBlobName(lbi)); }
  job_builder->AddOps(parallel_conf, {tick_op_conf});
  return tick_op_conf;
}

OperatorConf AppendTick(const std::list<OpNode*>& op_nodes, JobBuilder* job_builder) {
  std::vector<std::string> op_names;
  std::vector<LogicalBlobId> lbis;
  for (const auto* op_node : op_nodes) {
    if (op_node->op().output_bns().empty()) {
      op_names.push_back(op_node->op().op_name());
    } else {
      lbis.push_back(op_node->op().BnInOp2Lbi(op_node->op().output_bns().Get(0)));
    }
  }
  return AppendTick(op_names, lbis, op_nodes.front()->parallel_desc().parallel_conf(), job_builder);
}

OperatorConf PrependTick(const std::list<OpNode*>& op_nodes, JobBuilder* job_builder) {
  OperatorConf tick_op_conf = MakeTickOpConf();
  std::vector<OperatorConf> op_confs;
  for (const OpNode* op_node : op_nodes) {
    op_confs.push_back(op_node->op().op_conf());
    op_confs.back().add_ctrl_in_op_name(tick_op_conf.name());
  }
  job_builder->MutOps({op_confs});
  job_builder->AddOps(op_nodes.front()->parallel_desc().parallel_conf(), {tick_op_conf});
  return tick_op_conf;
}

void AppendAccTick(const Shape& src_shape, const std::list<OpNode*>& op_nodes,
                   JobBuilder* job_builder) {
  const auto& tick_shape = GetOpTimeShape(op_nodes.front());
  CHECK_EQ(tick_shape.elem_cnt() % src_shape.elem_cnt(), 0);
  const OperatorConf& tick_op_conf = AppendTick(op_nodes, job_builder);
  OperatorConf acc_op_conf;
  {
    acc_op_conf.set_name(std::string("System-AutoTick-Acc_") + NewUniqueId());
    auto* acc_conf = acc_op_conf.mutable_acc_conf();
    acc_conf->set_one(tick_op_conf.name() + "/" + tick_op_conf.tick_conf().out());
    acc_conf->set_acc("acc");
    acc_conf->set_max_acc_num(tick_shape.elem_cnt() / src_shape.elem_cnt());
  }
  OperatorConf last_tick_op_conf;
  {
    last_tick_op_conf.set_name(std::string("System-AutoTick-Tick_") + NewUniqueId());
    auto* tick_conf = last_tick_op_conf.mutable_tick_conf();
    tick_conf->add_tick(acc_op_conf.name() + "/acc");
    tick_conf->set_out("out");
  }
  job_builder->AddOps(op_nodes.front()->parallel_desc().parallel_conf(),
                      {acc_op_conf, last_tick_op_conf});
}

void AddGlobalCriticalSection(const std::string& src_tick_op_name,
                              const std::string& sink_tick_op_name, CriticalSectionType type) {
  auto critical_sec = std::make_unique<CriticalSection>();
  critical_sec->mutable_critical_section_id()->set_job_id(Global<JobDesc>::Get()->job_id());
  critical_sec->mutable_critical_section_id()->set_source_tick_op_name(src_tick_op_name);
  critical_sec->mutable_critical_section_id()->set_sink_tick_op_name(sink_tick_op_name);
  critical_sec->set_critical_section_type(type);
  Global<CriticalSectionDesc>::Get()->AddCriticalSection(std::move(critical_sec));
}

void ForEachInputOutputCriticalSectionOpNodes(
    const OpGraph& op_graph, const std::function<void(const HashSet<OpNode*>&)>& Handler) {
  HashMap<int32_t, HashSet<OpNode*>> op_type_case2op_nodes;
  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (op_conf.has_input_conf() || op_conf.has_output_conf() || op_conf.has_variable_conf()) {
      CHECK(op_type_case2op_nodes[op_conf.op_type_case()].emplace(op_node).second);
    }
  });
  for (OperatorConf::OpTypeCase op_type_case :
       {OperatorConf::kVariableConf, OperatorConf::kInputConf, OperatorConf::kOutputConf}) {
    if (op_type_case2op_nodes[op_type_case].empty() == false) {
      Handler(op_type_case2op_nodes[op_type_case]);
    }
  }
}

void AddGlobalInputOutputCriticalSection(const HashSet<OpNode*>& op_nodes, Job* job) {
  auto time_shape = std::make_unique<Shape>(std::vector<int64_t>{
      Global<JobDesc>::Get()->TotalBatchNum(), Global<JobDesc>::Get()->NumOfPiecesInBatch()});
  HashMap<ParallelDesc, std::list<OpNode*>> parallel_desc2op_nodes;
  for (OpNode* op_node : op_nodes) {
    parallel_desc2op_nodes[op_node->parallel_desc()].push_back(op_node);
    const auto& op_conf = op_node->op().op_conf();
    CHECK(op_conf.has_input_conf() || op_conf.has_output_conf() || op_conf.has_variable_conf());
    CHECK(*time_shape == *op_node->out_blob_time_shape());
  }
  JobBuilder job_builder(job);
  std::vector<OperatorConf> source_ticks;
  std::vector<OperatorConf> sink_ticks;
  for (const auto& pair : parallel_desc2op_nodes) {
    source_ticks.push_back(PrependTick(pair.second, &job_builder));
    sink_ticks.push_back(AppendTick(pair.second, &job_builder));
  }
  OperatorConf src_tick_op_conf;
  {
    BuildSourceTickOpAndParallelConf(&src_tick_op_conf, &job_builder);
    for (auto& op_conf : source_ticks) {
      op_conf.mutable_tick_conf()->add_tick(src_tick_op_conf.name() + "/"
                                            + src_tick_op_conf.source_tick_conf().out());
    }
    job_builder.MutOps(source_ticks);
  }
  OperatorConf sink_tick_op_conf;
  {
    BuildSinkTickOpAndParallelConf(&sink_tick_op_conf, &job_builder);
    for (const auto& op_conf : sink_ticks) {
      sink_tick_op_conf.mutable_sink_tick_conf()->add_tick(op_conf.name() + "/"
                                                           + op_conf.tick_conf().out());
    }
    job_builder.MutOps({sink_tick_op_conf});
  }
  AddGlobalCriticalSection(src_tick_op_conf.name(), sink_tick_op_conf.name(),
                           kInputOutputCriticalSection);
}

}  // namespace

void AutoSourceTick(const OpGraph& op_graph, Job* job) {
  GroupTickByParallelDesc(op_graph, job);
  op_graph.ForEachNode([&](OpNode* node) { CHECK(!node->op().op_conf().has_source_tick_conf()); });
  ConnectSourceTickAndOtherTick(job);
}

void AddTickForTimeShape(const OpGraph& op_graph, Job* job) {
  const auto& src_time_shape = *GetSrcTickOpNode(op_graph)->out_blob_time_shape();
  HashMap<std::pair<ParallelDesc, std::pair<Shape, Shape>>, std::list<OpNode*>> pd7ts2op_nodes;
  op_graph.ForEachNode([&](OpNode* op_node) {
    CHECK(!op_node->op().op_conf().has_sink_tick_conf());
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt > 0) { return; }
    auto ts = std::make_pair(*op_node->GetInputBlobFastestTimeShape(), GetOpTimeShape(op_node));
    pd7ts2op_nodes[{op_node->parallel_desc(), ts}].push_back(op_node);
  });
  JobBuilder job_builder(job);
  for (const auto& pair : pd7ts2op_nodes) {
    const std::pair<Shape, Shape>& ts = pair.first.second;
    if (ts.second == src_time_shape) {
      CHECK_GE(ts.first.elem_cnt(), ts.second.elem_cnt());
      AppendTick(pair.second, &job_builder);
    } else if (ts.second.elem_cnt() > src_time_shape.elem_cnt()) {
      AppendAccTick(src_time_shape, pair.second, &job_builder);
    } else {
      UNIMPLEMENTED();
    }
  }
}

void AutoSinkTick(const OpGraph& op_graph, Job* job) {
  op_graph.ForEachNode([&](OpNode* node) { CHECK(!node->op().op_conf().has_sink_tick_conf()); });
  const auto& src_time_shape = *GetSrcTickOpNode(op_graph)->out_blob_time_shape();
  HashSet<LogicalBlobId> tick_lbis;
  op_graph.ForEachNode([&](OpNode* op_node) {
    size_t out_cnt = 0;
    op_graph.ForEachDataAndCtrlOutNode(op_node, [&](OpNode*) { ++out_cnt; });
    if (out_cnt > 0) { return; }
    CHECK(op_node->op().op_conf().has_tick_conf());
    CHECK(*op_node->out_blob_time_shape() == src_time_shape);
    CHECK(tick_lbis.emplace(op_node->op().BnInOp2Lbi(op_node->op().SoleObn())).second);
  });
  JobBuilder job_builder(job);
  OperatorConf sink_tick_op_conf;
  BuildSinkTickOpAndParallelConf(&sink_tick_op_conf, &job_builder);
  for (const LogicalBlobId& tick_lbi : tick_lbis) {
    sink_tick_op_conf.mutable_sink_tick_conf()->add_tick(GenLogicalBlobName(tick_lbi));
  }
  job_builder.MutOps({sink_tick_op_conf});
}

void AddGlobalTotalJobCriticalSection(const Job& job) {
  const OperatorConf* src_tick_op_conf = nullptr;
  const OperatorConf* sink_tick_op_conf = nullptr;
  for (const auto& op_conf : job.net().op()) {
    if (op_conf.has_source_tick_conf()) {
      CHECK_ISNULL(src_tick_op_conf);
      src_tick_op_conf = &op_conf;
    }
    if (op_conf.has_sink_tick_conf()) {
      CHECK_ISNULL(sink_tick_op_conf);
      sink_tick_op_conf = &op_conf;
    }
  }
  CHECK_NOTNULL(src_tick_op_conf);
  CHECK_NOTNULL(sink_tick_op_conf);
  AddGlobalCriticalSection(src_tick_op_conf->name(), sink_tick_op_conf->name(),
                           kTotalJobCriticalSection);
}

void AddGlobalInputOutputCriticalSections(const OpGraph& op_graph, Job* job) {
  ForEachInputOutputCriticalSectionOpNodes(op_graph, [&](const HashSet<OpNode*>& op_nodes) {
    AddGlobalInputOutputCriticalSection(op_nodes, job);
  });
}

}  // namespace oneflow
