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

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_conf.pb.h"
#include <algorithm>

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/pass_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {

namespace {

using OpTypeSet = HashSet<std::string>;

const std::string FAKE_QUANT_SUFFIX = "-fake-quant";
const std::string ZP_SUFFIX = "-fake-quant-zp";
const std::string MOVING_MAX_SUFFIX = "-fake-quant-moving-max";
const std::string MOVING_MIN_SUFFIX = "-fake-quant-moving-min";
const std::string MUL_BIAS_SUFFIX = "-fake-quant-mul-bias";
const std::string OBSERVER_SUFFIX = "-fake-quant-observer";

void VerifyQATList(const OpTypeSet& op_list) {
  for (const auto& op_type : op_list) {
    CHECK(user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type) != nullptr)
        << "Cannot find " << op_type << " of QuantAwareTraining list in OpRegistry.";
  }
}

HashMap<std::string, std::string> scale_map;

Maybe<std::string> GetScaleLbn(const std::string& lbn) {
  CHECK_OR_RETURN(scale_map.find(lbn) != scale_map.end());
  return scale_map[lbn];
}

Maybe<bool> IsConvBiasEdge(const QatConfig& qat_config, const OpEdge* edge,
                           std::string* conv_input_scale_lbn, std::string* conv_weight_scale_lbn,
                           int64_t* weight_scale_length) {
  const auto* dst_node = edge->dst_node();

  const auto dst_op_type = dst_node->op().op_conf().user_conf().op_type_name();

  auto GetInputAndWeightScaleLbnAndWeightScaleLen4ConvNode =
      [](const QatConfig& qat_config, const OpNode* conv_node, std::string* conv_input_scale_lbn,
         std::string* conv_weight_scale_lbn, int64_t* weight_scale_length) -> Maybe<void> {
    *weight_scale_length = 1;
    for (const OpEdge* in_edge : conv_node->in_edges()) {
      CHECK_EQ_OR_RETURN(in_edge->lbis().size(), 1);
      const auto lbi = in_edge->lbis().front();
      const auto ibn = in_edge->lbi2ibns().at(lbi);
      CHECK_EQ_OR_RETURN(ibn.size(), 1);
      CHECK_OR_RETURN(ibn[0] == "in_0" || ibn[0] == "weight_0");
      if (ibn[0] == "in_0") {
        *conv_input_scale_lbn = *JUST(GetScaleLbn(GenLogicalBlobName(in_edge->lbis()[0])));
      } else if (ibn[0] == "weight_0") {
        if (qat_config.has_per_channel_weight_quantization()) {
          *weight_scale_length = conv_node->LogicalBlobDesc4Lbi(lbi).shape().At(0);
        }
        *conv_weight_scale_lbn = *JUST(GetScaleLbn(GenLogicalBlobName(in_edge->lbis()[0])));
      }
    }
    return Maybe<void>::Ok();
  };

  if (dst_op_type == "conv2d") {
    CHECK_EQ_OR_RETURN(edge->lbis().size(), 1);
    const auto lbi = edge->lbis().front();
    const auto ibn = edge->lbi2ibns().at(lbi);
    CHECK_EQ_OR_RETURN(ibn.size(), 1);
    if (ibn[0] == "bias_0") {
      JUST(GetInputAndWeightScaleLbnAndWeightScaleLen4ConvNode(
          qat_config, dst_node, conv_input_scale_lbn, conv_weight_scale_lbn, weight_scale_length));
      return true;
    }
  } else if (dst_op_type == "bias_add") {
    // check whether the bias_add corresponds to a conv
    for (const OpEdge* edge : dst_node->in_edges()) {
      const auto* src_node = edge->src_node();
      if (src_node->op().op_conf().user_conf().op_type_name() == "conv2d") {
        JUST(GetInputAndWeightScaleLbnAndWeightScaleLen4ConvNode(
            qat_config, src_node, conv_input_scale_lbn, conv_weight_scale_lbn,
            weight_scale_length));
        return true;
      }
    }
  }
  return false;
}

bool IsWeightEdge(const OpEdge* edge) {
  return edge->src_node()->op().op_conf().has_variable_conf();
}

bool IsBnInputEdge(const OpEdge* edge) {
  // Skip the inputs of bn for now.
  // In the complete qat pass, bn will be merged into conv.
  return edge->dst_node()->op().op_conf().user_conf().op_type_name() == "normalization";
}

std::string OpTypeName4OpNode(const OpNode* node) {
  return node->op().op_conf().user_conf().op_type_name();
}

using OpConfMap = HashMap<std::string, OperatorConf>;

OperatorConf Get1DZeroVariableOpConf(std::string name, const int64_t scope_symbol_id,
                                     const int64_t length, OpConfMap* inserted_ops) {
  OperatorConf variable_op_conf{};
  variable_op_conf.set_name(name);
  variable_op_conf.set_scope_symbol_id(scope_symbol_id);
  VariableOpConf* variable_conf = variable_op_conf.mutable_variable_conf();
  variable_conf->set_out("out");
  *variable_conf->mutable_shape()->mutable_dim()->Add() = length;
  variable_conf->set_data_type(DataType::kFloat);
  variable_conf->mutable_split_axis()->clear_value();
  variable_conf->mutable_initializer()->mutable_constant_conf()->set_value(0);
  (*inserted_ops)[name] = variable_op_conf;
  return variable_op_conf;
}

Maybe<OpNode*> GetInferenceOutputNode(const OpGraph& op_graph, OpNode* node) {
  OpNode* cur_node = node;
  if (node->op().op_conf().user_conf().op_type_name() == "conv2d"
      && node->out_edges().size() == 1) {
    OpNode* next_node = node->SoleOutEdge()->dst_node();
    if (OpTypeName4OpNode(next_node) == "bias_add") {
      cur_node = next_node;
      if (next_node->out_edges().size() == 1) { next_node = next_node->SoleOutEdge()->dst_node(); }
    }
    if (OpTypeName4OpNode(next_node) == "normalization") {
      cur_node = next_node;
      if (next_node->out_edges().size() == 1) { next_node = next_node->SoleOutEdge()->dst_node(); }
    }
    if (OpTypeName4OpNode(next_node) == "relu") { cur_node = next_node; }
  }
  VLOG(3) << "For node: " << node->op().op_name();
  VLOG(3) << "output node is: " << cur_node->op().op_name();
  return cur_node;
}

bool PerLayerQuantizationAttr4Config(const QatConfig& qat_config) {
  return !qat_config.per_channel_weight_quantization();
}

std::string QuantizationSchemeAttr4QatConfig(const QatConfig& qat_config) {
  return qat_config.symmetric() ? "symmetric" : "affine";
}

// TODO: refactor the following 4 methods by registration
std::string QuantizationFormulaAttr4QatConfig(const QatConfig& qat_config) {
  const auto target_backend = qat_config.target_backend();
  if (target_backend == "") {
    return "google";
  } else if (target_backend == "cambricon") {
    return "cambricon";
  } else {
    UNIMPLEMENTED();
  }
}

OpTypeSet Int8List4QatConfig(const QatConfig& qat_config) {
  const auto target_backend = qat_config.target_backend();
  if (target_backend == "") {
    return {"add_n", "matmul", "batch_matmul", "conv2d", "avg_pool_2d", "max_pool_2d"};
  } else if (target_backend == "cambricon") {
    return {"conv2d", "matmul"};
  } else {
    UNIMPLEMENTED();
  }
}

OpTypeSet TransparentList4QatConfig(const QatConfig& qat_config) {
  const auto target_backend = qat_config.target_backend();
  if (target_backend == "") {
    return {"reshape"};
  } else if (target_backend == "cambricon") {
    return {};
  } else {
    UNIMPLEMENTED();
  }
}

bool InsertQuantOpAfterInt8Ops4QatConfig(const QatConfig& qat_config) {
  const auto target_backend = qat_config.target_backend();
  if (target_backend == "") {
    return true;
  } else if (target_backend == "cambricon") {
    return false;
  } else {
    UNIMPLEMENTED();
  }
}

user_op::UserOpConfWrapper MultiplyOp(const std::string& name, const std::string& x,
                                      const std::string& y, const int64_t scope_symbol_id,
                                      OpConfMap* inserted_ops) {
  const auto op_wrapper = user_op::UserOpConfWrapperBuilder(name)
                              .Op("broadcast_mul")
                              .Input("x", x)
                              .Input("y", y)
                              .Output("z")
                              .ScopeSymbolId(scope_symbol_id)
                              .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

user_op::UserOpConfWrapper MinMaxObserver(const std::string& name, const std::string& input,
                                          const QatConfig& qat_config,
                                          const int64_t scope_symbol_id, OpConfMap* inserted_ops) {
  const auto op_wrapper =
      user_op::UserOpConfWrapperBuilder(name)
          .Op("min_max_observer")
          .Input("in", input)
          .Output("scale")
          .Output("zero_point")
          .Attr<std::string>("quantization_formula", QuantizationFormulaAttr4QatConfig(qat_config))
          .Attr<std::string>("quantization_scheme", QuantizationSchemeAttr4QatConfig(qat_config))
          .Attr("per_layer_quantization", PerLayerQuantizationAttr4Config(qat_config))
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

user_op::UserOpConfWrapper MovingMinMaxObserver(const std::string& name, const std::string& input,
                                                const std::string& train_step_lbn,
                                                const QatConfig& qat_config,
                                                const int64_t scope_symbol_id,
                                                OpConfMap* inserted_ops) {
  const std::string moving_max_name = name + MOVING_MAX_SUFFIX;
  const std::string moving_min_name = name + MOVING_MIN_SUFFIX;
  const auto moving_max_var =
      Get1DZeroVariableOpConf(moving_max_name, scope_symbol_id, 1, inserted_ops);
  const auto moving_min_var =
      Get1DZeroVariableOpConf(moving_min_name, scope_symbol_id, 1, inserted_ops);
  const auto op_wrapper =
      user_op::UserOpConfWrapperBuilder(name)
          .Op("moving_average_min_max_observer")
          .Input("in", input)
          .Input("current_train_step", train_step_lbn)
          .Input("moving_max",
                 GenLogicalBlobName(moving_max_var.name(), moving_max_var.variable_conf().out()))
          .Input("moving_min",
                 GenLogicalBlobName(moving_min_var.name(), moving_min_var.variable_conf().out()))
          .Output("scale")
          .Output("zero_point")
          .Attr("training", GlobalJobDesc().IsTrain())
          .Attr("stop_update_after_iters", qat_config.moving_min_max_stop_update_after_iters())
          .Attr<std::string>("quantization_formula", QuantizationFormulaAttr4QatConfig(qat_config))
          .Attr<std::string>("quantization_scheme", QuantizationSchemeAttr4QatConfig(qat_config))
          .Attr("momentum", qat_config.moving_min_max_momentum())
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

user_op::UserOpConfWrapper FakeQuantOp(const std::string& name, const std::string& input,
                                       const std::string& scale, const std::string& zero_point,
                                       const QatConfig& qat_config, const int64_t scope_symbol_id,
                                       OpConfMap* inserted_ops) {
  const auto op_wrapper =
      user_op::UserOpConfWrapperBuilder(name)
          .Op("fake_quantization")
          .Input("in", input)
          .Input("scale", scale)
          .Input("zero_point", zero_point)
          .Attr<std::string>("quantization_formula", QuantizationFormulaAttr4QatConfig(qat_config))
          .Attr<std::string>("quantization_scheme", QuantizationSchemeAttr4QatConfig(qat_config))
          .Output("out")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

Maybe<void> GetScaleAndZeroPointLbn4Edge(OpEdge* edge, const std::string train_step_lbn,
                                         std::string* scale, std::string* zero_point,
                                         const QatConfig& qat_config, const int64_t scope_symbol_id,
                                         OpConfMap* inserted_ops) {
  std::string lbn = GenLogicalBlobName(edge->lbis().front());
  std::string conv_input_scale_lbn;
  std::string conv_weight_scale_lbn;
  int64_t weight_scale_length;
  if (JUST(IsConvBiasEdge(qat_config, edge, &conv_input_scale_lbn, &conv_weight_scale_lbn,
                          &weight_scale_length))) {
    // mul scale
    const std::string mul_scale_op_name = ReplaceSlashToDash4Lbn(lbn) + MUL_BIAS_SUFFIX;
    CHECK_OR_RETURN(inserted_ops->find(mul_scale_op_name) == inserted_ops->end());
    const auto mul_scale_op = MultiplyOp(mul_scale_op_name, conv_input_scale_lbn,
                                         conv_weight_scale_lbn, scope_symbol_id, inserted_ops);

    *scale = mul_scale_op.output("z", 0);
    const std::string zp_var_name = ReplaceSlashToDash4Lbn(lbn) + ZP_SUFFIX;
    const auto zp_var =
        Get1DZeroVariableOpConf(zp_var_name, scope_symbol_id, weight_scale_length, inserted_ops);
    *zero_point = GenLogicalBlobName(zp_var.name(), zp_var.variable_conf().out());
  } else {
    const std::string observer_op_name = ReplaceSlashToDash4Lbn(lbn) + OBSERVER_SUFFIX;
    if (IsWeightEdge(edge)) {
      const auto observer_op =
          MinMaxObserver(observer_op_name, lbn, qat_config, scope_symbol_id, inserted_ops);
      *scale = observer_op.output("scale", 0);
      *zero_point = observer_op.output("zero_point", 0);
    } else {
      CHECK_OR_RETURN(qat_config.has_moving_min_max_stop_update_after_iters());
      const auto observer_op = MovingMinMaxObserver(observer_op_name, lbn, train_step_lbn,
                                                    qat_config, scope_symbol_id, inserted_ops);
      *scale = observer_op.output("scale", 0);
      *zero_point = observer_op.output("zero_point", 0);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReplaceInputLbn4DstNodeOfEdge(OpEdge* edge, const std::string& new_lbn,
                                          OpConfCache* op_conf_cache) {
  OpNode* dst_node = edge->dst_node();
  LogicalBlobId cur_lbi = edge->lbis().front();
  CHECK_EQ_OR_RETURN(1, edge->lbi2ibns().at(cur_lbi).size());
  const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

  OperatorConf dst_op_conf = op_conf_cache->GetLatest(dst_node->op().op_conf());
  ReplaceInputLbnInOpCustomizedConf(&dst_op_conf, dst_ibn, new_lbn);
  op_conf_cache->Put(dst_op_conf);
  return Maybe<void>::Ok();
}

class QuantAwareTraining final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(QuantAwareTraining);
  QuantAwareTraining() = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().job_conf().enable_quantization_aware_training();
  }

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;

 private:
  Maybe<void> InsertFakeQuantOp(const QatConfig& qat_config, const OpGraph& op_graph,
                                const OpTypeSet& int8_list, const OpTypeSet& transparent_list,
                                bool insert_quant_op_after_int8_ops,
                                HashSet<OpNode*> downstream_white, Job* job) const;
};

bool IsNodeQuantizationEnabled(const OpNode& node) {
  int64_t scope_symbol_id = node.op().op_conf().scope_symbol_id();
  CHECK(Global<symbol::Storage<Scope>>::Get()->Has(scope_symbol_id));
  const Scope& scope = Global<symbol::Storage<Scope>>::Get()->Get(scope_symbol_id);
  return scope.Bool("quantization_aware_training");
}

Maybe<void> QuantAwareTraining::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  CHECK(GlobalJobDesc().DefaultDataType() == DataType::kFloat);

  const auto qat_config = ctx->job_desc().job_conf().qat_config();

  OpTypeSet int8_list = Int8List4QatConfig(qat_config);
  OpTypeSet transparent_list = TransparentList4QatConfig(qat_config);
  // if `insert_quant_op_after_int8_ops` is false,
  // always insert quant op before int8 ops.
  // if `insert_quant_op_after_int8_ops` is true,
  // always insert quant op after int8 ops
  bool insert_quant_op_after_int8_ops = InsertQuantOpAfterInt8Ops4QatConfig(qat_config);

  VerifyQATList(int8_list);
  VerifyQATList(transparent_list);

  std::function<std::string(OpNode* const&)> OpName4Node = [](OpNode* const& node) {
    return node->op().op_name();
  };
  HashSet<OpNode*> white_set;
  DfsTopoGraphTraversal(op_graph, false,
                        [&int8_list](OpNode* node) { return IsNodeInList(int8_list, node); },
                        [&](OpNode* node) { return IsNodeInList(transparent_list, node); },
                        [&](OpNode* node) { return IsKeyFound(white_set, node); },
                        [&](OpNode* node) {
                          INSERT_CHECK(white_set.insert(node));
                          if (node->op().op_conf().user_conf().op_type_name() == "conv2d"
                              && node->out_edges().size() == 1) {
                            OpNode* next_node = node->SoleOutEdge()->dst_node();
                            if (OpTypeName4OpNode(next_node) == "bias_add") {
                              INSERT_CHECK(white_set.insert(next_node));
                              // TODO(daquexian): mark these special nodes
                              if (next_node->out_edges().size() == 1) {
                                next_node = next_node->SoleOutEdge()->dst_node();
                              }
                            }
                            if (OpTypeName4OpNode(next_node) == "normalization") {
                              INSERT_CHECK(white_set.insert(next_node));
                              if (next_node->out_edges().size() == 1) {
                                next_node = next_node->SoleOutEdge()->dst_node();
                              }
                            }
                            if (OpTypeName4OpNode(next_node) == "relu") {
                              INSERT_CHECK(white_set.insert(next_node));
                            }
                          }
                        });

  VLOG(3) << "white_set include: "
          << Container2Str<HashSet<OpNode*>, OpNode*>(white_set, OpName4Node);

  JUST(InsertFakeQuantOp(ctx->job_desc().job_conf().qat_config(), op_graph, int8_list,
                         transparent_list, insert_quant_op_after_int8_ops, white_set, job));
  return Maybe<void>::Ok();
}

// TODO: remove int8_list, transparent_list and insert_quant_op_after_int8_ops arguments
Maybe<void> QuantAwareTraining::InsertFakeQuantOp(const QatConfig& qat_config,
                                                  const OpGraph& op_graph,
                                                  const OpTypeSet& int8_list,
                                                  const OpTypeSet& transparent_list,
                                                  const bool insert_quant_op_after_int8_ops,
                                                  HashSet<OpNode*> white_set, Job* job) const {
  JobBuilder job_builder(job);
  HashSet<OpEdge*> white_set_edges;
  auto EdgeName4Edge = [](OpEdge* const& edge) {
    return std::string("edge of\t") + edge->src_node()->op().op_name() + "\tto\t"
           + edge->dst_node()->op().op_name();
  };
  auto AddWhiteSetEdge = [&white_set_edges, &EdgeName4Edge](OpEdge* edge) -> Maybe<void> {
    VLOG(3) << "insert " << EdgeName4Edge(edge);
    CHECK_EQ_OR_RETURN(edge->lbis().size(), 1);
    const std::string lbn = GenLogicalBlobName(edge->lbis().front());
    scale_map[lbn] = ReplaceSlashToDash4Lbn(lbn) + OBSERVER_SUFFIX + "/scale_0";
    VLOG(3) << "set " << lbn << " to " << scale_map[lbn];
    INSERT_CHECK(white_set_edges.insert(edge));
    return Maybe<void>::Ok();
  };
  auto PropagateScale = [](OpNode* node) -> Maybe<void> {
    CHECK_EQ_OR_RETURN(node->in_edges().size(), 1);
    CHECK_EQ_OR_RETURN(node->SoleInEdge()->lbis().size(), 1);
    for (OpEdge* edge : node->out_edges()) {
      CHECK_EQ_OR_RETURN(edge->lbis().size(), 1);
      const std::string node_input_lbn = GenLogicalBlobName(node->SoleInEdge()->lbis().front());
      const std::string lbn = GenLogicalBlobName(edge->lbis().front());
      if (scale_map.find(node_input_lbn) != scale_map.end()) {
        scale_map[lbn] = scale_map[node_input_lbn];
      }
    }
    return Maybe<void>::Ok();
  };

  {
    JUST(op_graph.MaybeForEachNode([&](OpNode* node) -> Maybe<void> {
      if (IsKeyFound(white_set, node)) {
        for (OpEdge* edge : node->in_edges()) {
          if (IsKeyFound(white_set, edge->src_node())) { continue; }
          if (IsNodeQuantizationEnabled(*edge->dst_node())) { JUST(AddWhiteSetEdge(edge)); }
        }
        if (IsNodeInList(int8_list, node)) {
          if (insert_quant_op_after_int8_ops) {
            OpNode* inference_node = JUST(GetInferenceOutputNode(op_graph, node));
            if (IsNodeQuantizationEnabled(*inference_node)) {
              for (OpEdge* edge : inference_node->out_edges()) { JUST(AddWhiteSetEdge(edge)); }
            }
          } else {
            if (IsNodeQuantizationEnabled(*node)) {
              for (OpEdge* edge : node->in_edges()) {
                if (white_set_edges.find(edge) == white_set_edges.end()) {
                  JUST(AddWhiteSetEdge(edge));
                }
              }
            }
          }
        } else if (IsNodeInList(transparent_list, node)) {
          JUST(PropagateScale(node));
        } else {
          // this is bias_add, relu or bn op in "conv -> bias_add -> bn -> relu" pattern,
          // do nothing
        }
      }
      return Maybe<void>::Ok();
    }));
    VLOG(3) << "white_set_edges: "
            << Container2Str<HashSet<OpEdge*>, OpEdge*>(white_set_edges, EdgeName4Edge);
  }

  // group edges by lbn so that we can use `src_node` when calling `AddOps`
  HashMap<std::string, std::vector<OpEdge*>> edges_group_by_lbn;
  {
    for (OpEdge* edge : white_set_edges) {
      CHECK_EQ_OR_RETURN(1, edge->lbis().size());
      std::string lbn = GenLogicalBlobName(edge->lbis().front());
      edges_group_by_lbn[lbn].push_back(edge);
    }
  }

  OpConfCache op_conf_cache;
  for (auto& pair : edges_group_by_lbn) {
    const std::string& lbn = pair.first;
    const OpNode* src_node = pair.second.front()->src_node();

    const BlobDesc& blob_desc = src_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(lbn));
    if (blob_desc.data_type() != DataType::kFloat) { continue; }

    OpConfMap inserted_ops;
    for (OpEdge* edge : pair.second) {
      if (IsBnInputEdge(edge)) { continue; }
      std::string scale;
      std::string zero_point;
      const int64_t scope_symbol_id = edge->src_node()->op().op_conf().scope_symbol_id();
      JUST(GetScaleAndZeroPointLbn4Edge(edge, job->job_conf().train_conf().train_step_lbn(), &scale,
                                        &zero_point, qat_config, scope_symbol_id, &inserted_ops));
      const std::string fake_quant_op_name = ReplaceSlashToDash4Lbn(lbn) + FAKE_QUANT_SUFFIX;
      const auto fake_quant_op = FakeQuantOp(fake_quant_op_name, lbn, scale, zero_point, qat_config,
                                             scope_symbol_id, &inserted_ops);

      const std::string fake_quant_op_output_name = fake_quant_op.output("out", 0);

      JUST(ReplaceInputLbn4DstNodeOfEdge(edge, fake_quant_op_output_name, &op_conf_cache));
    }

    for (const auto& pair : inserted_ops) {
      VLOG(3) << "Insert op: " << pair.second.DebugString() << " between " << lbn;
      job_builder.AddOps(src_node->parallel_desc().parallel_conf(), {pair.second});
    }
  }

  job_builder.MutOpsOnlyOnce(op_conf_cache.op_confs());
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("QuantAwareTraining", QuantAwareTraining);

}  // namespace

}  // namespace oneflow
