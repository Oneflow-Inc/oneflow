#include "oneflow/core/job_completer/all_reduce_add_pass.h"

namespace oneflow {

namespace {

std::function<const OpNode*(const LogicalBlobId&)> MakeGetterProducerOpNode4Lbi(
    const OpGraph& op_graph) {
  auto lbi2producer_op_node = std::make_shared<HashMap<LogicalBlobId, const OpNode*>>();
  op_graph.ForEachNode([&](OpNode* op_node) {
    for (const std::string& obn : op_node->op().output_bns()) {
      CHECK(lbi2producer_op_node->emplace(op_node->op().BnInOp2Lbi(obn), op_node).second);
    }
  });
  return [lbi2producer_op_node](const LogicalBlobId& lbi) -> const OpNode* {
    return lbi2producer_op_node->at(lbi);
  };
}

std::function<const OpNode*(const LogicalBlobId&)> MakeGetterSoleConsumerOpNode4Lbi(
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi) {
  return [ProducerOpNode4Lbi](const LogicalBlobId& lbi) -> const OpNode* {
    const OpNode* producer = ProducerOpNode4Lbi(lbi);
    const OpEdge* consumer_edge = nullptr;
    for (const OpEdge* edge : producer->out_edges()) {
      if (std::find(edge->lbis().begin(), edge->lbis().end(), lbi) != edge->lbis().end()) {
        CHECK(consumer_edge == nullptr);
        consumer_edge = edge;
      }
    }
    CHECK_NOTNULL(consumer_edge);
    return consumer_edge->dst_node();
  };
}

LogicalBlobId FindP2BLbiWithSoleConsumer(
    const LogicalBlobId& lbi,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::function<const OpNode*(const LogicalBlobId&)>& SoleConsumerOpNode4Lbi) {
  CHECK(SoleConsumerOpNode4Lbi(lbi)->SbpParallel4Lbi(lbi).has_broadcast_parallel());
  LogicalBlobId cur_lbi = lbi;
  auto SoleBodyIbn4Op = [&](const Operator& op) -> const std::string& {
    if (op.input_bns().size() == 1) {
      return op.SoleIbn();
    } else if (op.input_bns().size() > 1) {
      const std::string* data_bn = nullptr;
      for (const auto& ibn : op.input_bns()) {
        if (op.InputBlobModifier4Ibn(ibn).use_header_only() == false) {
          CHECK(data_bn == nullptr);
          data_bn = &ibn;
        }
      }
      CHECK_NOTNULL(data_bn);
      return *data_bn;
    } else {
      UNIMPLEMENTED();
    }
  };
  while (!ProducerOpNode4Lbi(cur_lbi)->SbpParallel4Lbi(cur_lbi).has_partial_sum_parallel()) {
    const auto* producer = ProducerOpNode4Lbi(cur_lbi);
    CHECK_EQ(producer->op().output_bns().size(), 1);
    LogicalBlobId in_lbi = producer->op().BnInOp2Lbi(SoleBodyIbn4Op(producer->op()));
    CHECK(producer->SbpParallel4Lbi(cur_lbi).has_broadcast_parallel());
    CHECK(producer->SbpParallel4Lbi(in_lbi).has_broadcast_parallel());
    cur_lbi = in_lbi;
  }
  return cur_lbi;
}

void FindAllReducedLbis(
    const Job& job, const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<LogicalBlobId>* lbis) {
  auto SoleConsumerOpNode4Lbi = MakeGetterSoleConsumerOpNode4Lbi(ProducerOpNode4Lbi);
  const auto& lbi2diff_lbi = job.helper().tag2lbi_relations().at(kProducedLbi2ConsumedDiffLbi);
  HashSet<LogicalBlobId> key_check;
  HashSet<LogicalBlobId> diff_lbis;
  for (const auto& pair : lbi2diff_lbi.pair()) {
    CHECK(key_check.emplace(pair.first()).second);
    const auto* producer = ProducerOpNode4Lbi(pair.first());
    if (producer->parallel_desc().parallel_num() == 1) { continue; }
    if (producer->op().op_conf().has_variable_conf() == false) { continue; }
    if (producer->SbpParallel4Lbi(pair.first()).has_broadcast_parallel() == false) { continue; }
    const auto& diff_lbi =
        FindP2BLbiWithSoleConsumer(pair.second(), ProducerOpNode4Lbi, SoleConsumerOpNode4Lbi);
    diff_lbis.insert(diff_lbi);
  }
  *lbis = {diff_lbis.begin(), diff_lbis.end()};
}

std::function<int32_t(const OpNode*)> MakeGetterDepth4OpNode(const OpGraph& op_graph) {
  auto op_node2depth = std::make_shared<HashMap<const OpNode*, int32_t>>();
  int32_t depth = 0;
  op_graph.TopoForEachNode([&](OpNode* op_node) { op_node2depth->emplace(op_node, depth++); });
  return [op_node2depth](const OpNode* op_node) { return op_node2depth->at(op_node); };
}

void SortAllReducedLbis(
    const OpGraph& op_graph,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    std::vector<LogicalBlobId>* lbis) {
  auto Depth4OpNode = MakeGetterDepth4OpNode(op_graph);
  std::sort(lbis->begin(), lbis->end(), [&](const LogicalBlobId& lhs, const LogicalBlobId& rhs) {
    return Depth4OpNode(ProducerOpNode4Lbi(lhs)) > Depth4OpNode(ProducerOpNode4Lbi(rhs));
  });
}

void ForEachLbisGroupByDataTypeAndParallelDesc(
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& sorted_lbis,
    const std::function<void(const std::vector<LogicalBlobId>&)>& Handler) {
  HashMap<int32_t, HashMap<ParallelDesc, std::vector<LogicalBlobId>>> dtype2parellel_desc2op_nodes;
  for (const LogicalBlobId& lbi : sorted_lbis) {
    const OpNode* producer = ProducerOpNode4Lbi(lbi);
    DataType dtype = producer->LogicalBlobDesc4Lbi(lbi).data_type();
    dtype2parellel_desc2op_nodes[dtype][producer->parallel_desc()].push_back(lbi);
  }
  for (const auto& out_pair : dtype2parellel_desc2op_nodes) {
    for (const auto& pair : out_pair.second) { Handler(pair.second); }
  }
}

void GroupAllReducedLbisByStrategy(
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& sorted_lbis,
    std::vector<std::vector<LogicalBlobId>>* lbi_groups) {
  auto MemSize4Lbi = [&](const LogicalBlobId& lbi) -> size_t {
    const OpNode* producer = ProducerOpNode4Lbi(lbi);
    const BlobDesc& logical_blob_desc = producer->LogicalBlobDesc4Lbi(lbi);
    int64_t elem_cnt = logical_blob_desc.shape().elem_cnt();
    size_t model_size = elem_cnt * GetSizeOfDataType(logical_blob_desc.data_type());
    return RoundUp(model_size, kCudaAlignSize);
  };
  ForEachLbisGroupByDataTypeAndParallelDesc(
      ProducerOpNode4Lbi, sorted_lbis, [&](const std::vector<LogicalBlobId>& lbis) {
        size_t model_total_size = 0;
        for (const auto& lbi : lbis) { model_total_size += MemSize4Lbi(lbi); }
        size_t avg_size = model_total_size / GlobalJobDesc().all_reduce_group_num();
        const size_t group_min_size = GlobalJobDesc().all_reduce_group_min_byte();
        const float group_size_warmup = GlobalJobDesc().all_reduce_group_size_warmup();
        size_t cur_group_capacity = group_min_size / group_size_warmup;
        size_t cur_group_model_size = GetMaxVal<size_t>();
        for (const LogicalBlobId& lbi : lbis) {
          if (cur_group_model_size >= cur_group_capacity) {
            lbi_groups->emplace_back(std::vector<LogicalBlobId>{});
            cur_group_model_size = 0;
            if (cur_group_capacity < avg_size) { cur_group_capacity *= group_size_warmup; }
          }
          lbi_groups->back().emplace_back(lbi);
          cur_group_model_size += MemSize4Lbi(lbi);
        }
      });
}

void AddReduceConcatAndReduceIdentityOpConf(JobBuilder* job_builder,
                                            const ParallelConf& parallel_conf,
                                            const std::vector<LogicalBlobId>& lbi_groups,
                                            int32_t order_in_graph, LogicalBlobId* grouped_lbi) {
  OperatorConf reduce_concat_op_conf;
  reduce_concat_op_conf.set_name("System-Boxing-AllReduce-ReduceConcat_" + NewUniqueId());
  auto* reduce_concat_conf = reduce_concat_op_conf.mutable_reduce_concat_conf();
  reduce_concat_conf->set_in_num(lbi_groups.size());
  for (const LogicalBlobId& lbi : lbi_groups) {
    reduce_concat_conf->add_in(GenLogicalBlobName(lbi));
  }
  reduce_concat_conf->set_out("out");

  OperatorConf reduce_identity_op_conf;
  reduce_identity_op_conf.set_name("System-Boxing-AllReduce-ReduceIdentity_" + NewUniqueId());
  auto* reduce_identity_conf = reduce_identity_op_conf.mutable_reduce_identity_conf();
  reduce_identity_conf->set_in(reduce_concat_op_conf.name() + "/out");
  reduce_identity_conf->set_out("out");
  reduce_identity_conf->set_order_in_graph(order_in_graph);
  job_builder->AddOps(parallel_conf, {reduce_concat_op_conf, reduce_identity_op_conf});
  *grouped_lbi = GenLogicalBlobId(reduce_identity_op_conf.name() + "/out");
}

void AddAllReduceOpConf(JobBuilder* job_builder, const ParallelConf& parallel_conf,
                        const LogicalBlobId& grouped_lbi, LogicalBlobId* all_reduced_lbi) {
  OperatorConf all_reduce_op{};
  all_reduce_op.set_name("System-Boxing-AllReduce-" + grouped_lbi.op_name() + "-"
                         + grouped_lbi.blob_name());
  AllReduceFacadeOpConf* all_reduce_facade_op_conf = all_reduce_op.mutable_all_reduce_facade_conf();
  all_reduce_facade_op_conf->set_in(GenLogicalBlobName(grouped_lbi));
  all_reduce_facade_op_conf->set_out("out");
  all_reduced_lbi->set_op_name(all_reduce_op.name());
  all_reduced_lbi->set_blob_name(all_reduce_facade_op_conf->out());
  job_builder->AddOps(parallel_conf, {all_reduce_op});
}

void AddReduceSplitOpConf(
    JobBuilder* job_builder,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& lbi_groups, int32_t order_in_graph,
    const LogicalBlobId& all_reduced_lbi) {
  auto SoleConsumerOpNode4Lbi = MakeGetterSoleConsumerOpNode4Lbi(ProducerOpNode4Lbi);
  auto MutModelUpdateOpConf = [&](const LogicalBlobId& lbi, const LogicalBlobId& new_lbi) {
    const auto* op_node = SoleConsumerOpNode4Lbi(lbi);
    OperatorConf md_updt_op_conf(op_node->op().op_conf());
    std::string ibn = "";
    for (const auto& bn : op_node->op().input_bns()) {
      if (op_node->op().BnInOp2Lbi(bn) == lbi) {
        CHECK(ibn == "");
        ibn = bn;
      }
    }
    CHECK(ibn != "");
    PbMessage* md_updt_conf =
        MutableMessageInPbMessage(&md_updt_op_conf, md_updt_op_conf.op_type_case());
    ReplaceStrValInPbFdOrPbRpf(md_updt_conf, ibn, GenLogicalBlobName(lbi),
                               GenLogicalBlobName(new_lbi));
    job_builder->MutOpsOnlyOnce({md_updt_op_conf});
  };

  OperatorConf reduce_split_op_conf;
  reduce_split_op_conf.set_name("System-Boxing-AllReduce-ReduceSplit_" + NewUniqueId());
  auto* reduce_split_conf = reduce_split_op_conf.mutable_reduce_split_conf();
  reduce_split_conf->set_in(GenLogicalBlobName(all_reduced_lbi));
  reduce_split_conf->set_out_num(lbi_groups.size());
  reduce_split_conf->set_order_in_graph(order_in_graph);
  FOR_RANGE(int32_t, i, 0, lbi_groups.size()) {
    const LogicalBlobId& lbi = lbi_groups.at(i);
    const std::string& out_blob_name = std::string("out_") + std::to_string(i);
    reduce_split_conf->add_out(out_blob_name);
    ProducerOpNode4Lbi(lbi)->LogicalBlobDesc4Lbi(lbi).shape().ToProto(
        reduce_split_conf->add_out_shape());
    MutModelUpdateOpConf(lbi, GenLogicalBlobId(reduce_split_op_conf.name() + "/" + out_blob_name));
  }
  job_builder->AddOps(ProducerOpNode4Lbi(lbi_groups.at(0))->parallel_desc().parallel_conf(),
                      {reduce_split_op_conf});
}

void BuildAllReduceStruct(
    JobBuilder* job_builder,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& lbi_groups, int32_t order_in_graph) {
  const auto& parallel_conf = ProducerOpNode4Lbi(lbi_groups.at(0))->parallel_desc().parallel_conf();
  LogicalBlobId grouped_lbi;
  AddReduceConcatAndReduceIdentityOpConf(job_builder, parallel_conf, lbi_groups, order_in_graph,
                                         &grouped_lbi);
  LogicalBlobId all_reduced_lbi;
  AddAllReduceOpConf(job_builder, parallel_conf, grouped_lbi, &all_reduced_lbi);
  AddReduceSplitOpConf(job_builder, ProducerOpNode4Lbi, lbi_groups, order_in_graph,
                       all_reduced_lbi);
}

}  // namespace

void AllReduceAddPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  auto ProducerOpNode4Lbi = MakeGetterProducerOpNode4Lbi(op_graph);

  std::vector<LogicalBlobId> lbis;
  FindAllReducedLbis(job_builder->job(), op_graph, ProducerOpNode4Lbi, &lbis);
  SortAllReducedLbis(op_graph, ProducerOpNode4Lbi, &lbis);
  HashMap<LogicalBlobId, int32_t> lbi2order_in_graph;
  FOR_RANGE(int32_t, i, 0, lbis.size()) { CHECK(lbi2order_in_graph.emplace(lbis.at(i), i).second); }

  std::vector<std::vector<LogicalBlobId>> lbi_groups;
  GroupAllReducedLbisByStrategy(ProducerOpNode4Lbi, lbis, &lbi_groups);
  FOR_RANGE(int32_t, i, 0, lbi_groups.size()) {
    const auto& lbi_group = lbi_groups.at(i);
    BuildAllReduceStruct(job_builder, ProducerOpNode4Lbi, lbi_group,
                         lbi2order_in_graph.at(lbi_group.at(0)));
  }
}

}  // namespace oneflow
