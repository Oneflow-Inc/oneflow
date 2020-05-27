#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

class HalfBoxingPass final : public OpGraphPass {
 public:
  HalfBoxingPass() = default;
  ~HalfBoxingPass() override = default;
  bool IsEnabled() const override { return GlobalJobDesc().use_boxing_v2(); }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

void HalfBoxingPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  op_graph.ForEachNode([&job_builder](OpNode* op_node) {
    // find cast_fp16_to_fp32 -> all_reduce(boxing) pattern
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_collective_boxing_generic_conf()) { return; }
    auto* prev_node = op_node->SoleInEdge()->src_node();
    if (prev_node->out_edges().size() != 1) { return; }
    auto prev_op_conf = prev_node->op().op_conf();
    if (!(prev_op_conf.has_user_conf() && prev_op_conf.user_conf().op_type_name() == "Cast")) {
      return;
    }
    auto prev_op_cast_conf = prev_op_conf.user_conf();
    if (prev_op_cast_conf.attr().at("dtype").at_data_type() == DataType::kFloat) { return; }

    // replace boxing op input with cast op input
    {
      auto new_op_conf = op_conf;
      auto boxing_conf = new_op_conf.collective_boxing_generic_conf();
      std::string cast_input = prev_op_cast_conf.input().at("in").s(0);
      std::string boxing_input = boxing_conf.lbi().blob_name();
      ReplaceInputLbnInOpCustomizedConf(&boxing_conf, "lbi", boxing_input, cast_input);
      job_builder->MutOpsOnlyOnce({new_op_conf});
    }
    // replace cast op input with boxing op output
    {
      auto new_prev_op_conf = prev_op_conf;
      std::string boxing_output = op_conf.collective_boxing_generic_conf().lbi().blob_name();
      std::string cast_input = prev_op_cast_conf.input().at("in").s(0);
      ReplaceInputLbnInOpCustomizedConf(new_prev_op_conf.mutable_user_conf(), "in_0", cast_input,
                                        boxing_output);
      job_builder->MutOpsOnlyOnce({new_prev_op_conf});
    }

    // update all boxing op consumers
    for (OpEdge* edge : op_node->out_edges()) {
      OpNode* dst_node = edge->dst_node();
      LogicalBlobId cur_lbi = edge->lbis().front();
      const auto lbn = GenLogicalBlobName(cur_lbi);
      CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
      const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

      OperatorConf dst_op_conf = dst_node->op().op_conf();
      PbMessage* dst_op_type_conf =
          MutableMessageInPbMessage(&dst_op_conf, dst_op_conf.op_type_case());
      std::string cast_output = prev_op_cast_conf.output().at("out").s(0);
      // if (!TryUpdtBnVal4SepcialOpConf(dst_op_conf.op_type_case(), dst_op_type_conf, lbn, new_lbn,
      // dst_ibn)) {
      ReplaceInputLbnInOpCustomizedConf(dst_op_type_conf, dst_ibn, lbn, cast_output);
      // }
      job_builder->MutOpsOnlyOnce({dst_op_conf});
    }
  });
}

}  // namespace

REGISTER_FUNCTION_PASS("HalfBoxingPass", HalfBoxingPass);

}  // namespace oneflow
