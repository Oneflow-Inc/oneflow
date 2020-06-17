#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

class OpConfCache {
  std::map<std::string, OperatorConf> _op_confs_to_update;

 public:
  OperatorConf GetLatest(const OperatorConf& op_conf) {
    if (_op_confs_to_update.find(op_conf.name()) != _op_confs_to_update.end()) {
      return _op_confs_to_update[op_conf.name()];
    }
    return op_conf;
  }
  void Put(const OperatorConf& op_conf) { _op_confs_to_update[op_conf.name()] = op_conf; }
  std::vector<OperatorConf> op_confs() {
    std::vector<OperatorConf> ret;
    for (const auto& x : _op_confs_to_update) { ret.push_back(x.second); }
    return ret;
  }
};

class DoParallelCastBeforeWideningTypeCast final : public OpGraphPass {
 public:
  DoParallelCastBeforeWideningTypeCast() = default;
  ~DoParallelCastBeforeWideningTypeCast() override = default;
  bool IsEnabled() const override {
    return GlobalJobDesc().do_parallel_cast_before_widening_type_cast()
           && GlobalJobDesc().use_boxing_v2();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

Maybe<void> DoParallelCastBeforeWideningTypeCast::Apply(const OpGraph& op_graph,
                                                        JobBuilder* job_builder) const {
  OpConfCache op_conf_cache;
  op_graph.ForEachNode([&op_conf_cache](OpNode* parallel_cast_node) {
    // find cast_fp16_to_fp32_or_double -> parallel_cast pattern
    const OperatorConf& parallel_cast_op_conf =
        op_conf_cache.GetLatest(parallel_cast_node->op().op_conf());
    if (!parallel_cast_op_conf.has_parallel_cast_conf()) { return; }
    auto* cast_node = parallel_cast_node->SoleInEdge()->src_node();
    if (cast_node->out_edges().size() != 1) { return; }
    auto cast_op_conf = op_conf_cache.GetLatest(cast_node->op().op_conf());
    if (!(cast_op_conf.has_user_conf() && cast_op_conf.user_conf().op_type_name() == "cast")) {
      return;
    }
    const auto cast_conf_wrapper = user_op::UserOpConfWrapper(cast_op_conf);
    const auto cast_in_lbi = cast_node->SoleInEdge()->lbis().front();
    const auto cast_in_dtype = cast_node->LogicalBlobDesc4Lbi(cast_in_lbi).data_type();
    const auto cast_out_dtype = cast_conf_wrapper.attr<DataType>("dtype");
    if (!(cast_in_dtype == DataType::kFloat16
          && (cast_out_dtype == DataType::kFloat || cast_out_dtype == DataType::kDouble))) {
      return;
    }

    // replace parallel_cast op input with cast op input
    {
      auto new_parallel_cast_op_conf = parallel_cast_op_conf;
      const std::string cast_input = cast_conf_wrapper.input("in", 0);
      const std::string parallel_cast_input = parallel_cast_op_conf.parallel_cast_conf().in();
      ReplaceInputLbnInOpCustomizedConf(new_parallel_cast_op_conf.mutable_parallel_cast_conf(),
                                        "in", parallel_cast_input, cast_input);

      op_conf_cache.Put(new_parallel_cast_op_conf);
    }
    // replace cast op input with parallel_cast op output
    {
      auto new_cast_op_conf = cast_op_conf;
      const std::string parallel_cast_output =
          parallel_cast_op_conf.name() + "/" + parallel_cast_op_conf.parallel_cast_conf().out();
      const std::string cast_input = cast_conf_wrapper.input("in", 0);
      ReplaceInputLbnInOpCustomizedConf(new_cast_op_conf.mutable_user_conf(), "in_0", cast_input,
                                        parallel_cast_output);
      op_conf_cache.Put(new_cast_op_conf);
    }

    // update all parallel_cast op consumers
    const std::string cast_output = cast_conf_wrapper.output("out", 0);
    for (OpEdge* edge : parallel_cast_node->out_edges()) {
      CHECK_EQ(1, edge->lbis().size());
      LogicalBlobId cur_lbi = edge->lbis().front();
      const auto lbn = GenLogicalBlobName(cur_lbi);
      CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
      const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

      OpNode* dst_node = edge->dst_node();
      OperatorConf dst_op_conf = op_conf_cache.GetLatest(dst_node->op().op_conf());
      PbMessage* dst_op_type_conf =
          MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
      ReplaceInputLbnInOpCustomizedConf(dst_op_type_conf, dst_ibn, lbn, cast_output);
      op_conf_cache.Put(dst_op_conf);
    }
  });
  job_builder->MutOpsOnlyOnce(op_conf_cache.op_confs());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_FUNCTION_PASS("DoParallelCastBeforeWideningTypeCast",
                       DoParallelCastBeforeWideningTypeCast);

}  // namespace oneflow
