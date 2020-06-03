#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

class HalfBoxingPass final : public OpGraphPass {
 public:
  HalfBoxingPass() = default;
  ~HalfBoxingPass() override = default;
  bool IsEnabled() const override { return GlobalJobDesc().IsTrain(); }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

void HalfBoxingPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  op_graph.ForEachNode([&job_builder](OpNode* parallel_cast_node) {
    // find cast_fp16_to_fp32_or_double -> parallel_cast pattern
    const OperatorConf& parallel_cast_op_conf = parallel_cast_node->op().op_conf();
    if (!parallel_cast_op_conf.has_parallel_cast_conf()) { return; }
    auto* cast_node = parallel_cast_node->SoleInEdge()->src_node();
    if (cast_node->out_edges().size() != 1) { return; }
    auto cast_op_conf = cast_node->op().op_conf();
    if (!(cast_op_conf.has_user_conf() && cast_op_conf.user_conf().op_type_name() == "cast")) {
      return;
    }
    const auto cast_conf = cast_op_conf.user_conf();
    const auto cast_in_lbi = cast_node->SoleInEdge()->lbis().front();
    const auto cast_in_dtype = cast_node->LogicalBlobDesc4Lbi(cast_in_lbi).data_type();
    const auto cast_out_dtype = cast_conf.attr().at("dtype").at_data_type();
    if (!(cast_in_dtype == DataType::kFloat16
          && (cast_out_dtype == DataType::kFloat || cast_out_dtype == DataType::kDouble))) {
      return;
    }

    // replace parallel_cast op input with cast op input
    {
      auto new_parallel_cast_op_conf = parallel_cast_op_conf;
      const std::string cast_input = cast_conf.input().at("in").s(0);
      const std::string parallel_cast_input = parallel_cast_op_conf.parallel_cast_conf().in();
      ReplaceInputLbnInOpCustomizedConf(new_parallel_cast_op_conf.mutable_parallel_cast_conf(),
                                        "in", parallel_cast_input, cast_input);
      job_builder->MutOpsOnlyOnce({new_parallel_cast_op_conf});
    }
    // replace cast op input with parallel_cast op output
    {
      auto new_cast_op_conf = cast_op_conf;
      const std::string parallel_cast_output =
          parallel_cast_op_conf.name() + "/" + parallel_cast_op_conf.parallel_cast_conf().out();
      const std::string cast_input = cast_conf.input().at("in").s(0);
      ReplaceInputLbnInOpCustomizedConf(new_cast_op_conf.mutable_user_conf(), "in_0", cast_input,
                                        parallel_cast_output);
      job_builder->MutOpsOnlyOnce({new_cast_op_conf});
    }

    // update all parallel_cast op consumers
    const std::string cast_output = cast_conf.output().at("out").s(0);
    for (OpEdge* edge : parallel_cast_node->out_edges()) {
      CHECK_EQ(1, edge->lbis().size());
      LogicalBlobId cur_lbi = edge->lbis().front();
      const auto lbn = GenLogicalBlobName(cur_lbi);
      CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
      const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

      OpNode* dst_node = edge->dst_node();
      OperatorConf dst_op_conf = dst_node->op().op_conf();
      PbMessage* dst_op_type_conf =
          MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
      ReplaceInputLbnInOpCustomizedConf(dst_op_type_conf, dst_ibn, lbn, cast_output);
      job_builder->MutOpsOnlyOnce({dst_op_conf});
    }
  });
}

}  // namespace

REGISTER_FUNCTION_PASS("HalfBoxingPass", HalfBoxingPass);

}  // namespace oneflow
