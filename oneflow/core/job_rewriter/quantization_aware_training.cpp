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
#ifdef WITH_CUDA

#include "oneflow/core/job_rewriter/auto_mixed_precision_lists.h"

#include <algorithm>

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/pass_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

using QATList = HashSet<std::string>;

const QATList& Int8List() {
  static QATList int8_list = {"matmul", "batch_matmul", "conv2d", "avg_pool_2d", "max_pool_2d"};
  return int8_list;
}

const QATList& TransparentList() {
  static QATList transparent_list = {
      "reshape",
  };
  return transparent_list;
}

const std::string fake_quant_suffix = "-fake-quant";
const std::string zp_suffix = "-fake-quant-zp";
const std::string moving_max_suffix = "-fake-quant-moving-max";
const std::string moving_min_suffix = "-fake-quant-moving-min";
const std::string mul_bias_suffix = "-fake-quant-mul-bias";
const std::string observer_suffix = "-fake-quant-observer";

namespace {

void VerifyQATList(const QATList& amp_list) {
  for (const auto& op_type : amp_list) {
    CHECK(user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type) != nullptr)
        << "Cannot find " << op_type << " of QuantAwareTraining list in OpRegistry.";
  }
}

HashMap<std::string, std::string> scale_map;

std::string GetScaleLbn(const std::string& lbn) {
  assert(scale_map.find(lbn) != scale_map.end());
  return scale_map[lbn];
}

bool IsConvBiasEdge(const OpEdge* edge, std::string* conv_input_scale_lbn,
                    std::string* conv_weight_scale_lbn) {
  const auto* dst_node = edge->dst_node();
  if (dst_node->op().op_conf().user_conf().op_type_name() != "bias_add") { return false; }
  for (const OpEdge* edge : dst_node->in_edges()) {
    const auto* src_node = edge->src_node();
    if (src_node->op().op_conf().user_conf().op_type_name() == "conv2d") {
      for (const OpEdge* in_edge : src_node->in_edges()) {
        assert(in_edge->lbis().size() == 1);
        const auto lbi = in_edge->lbis().front();
        const auto ibn = in_edge->lbi2ibns().at(lbi);
        assert(ibn.size() == 1);
        assert(ibn[0] == "in_0" || ibn[0] == "weight_0");
        if (ibn[0] == "in_0") {
          *conv_input_scale_lbn = GetScaleLbn(GenLogicalBlobName(in_edge->lbis()[0]));
        } else if (ibn[0] == "weight_0") {
          *conv_weight_scale_lbn = GetScaleLbn(GenLogicalBlobName(in_edge->lbis()[0]));
        }
      }
      return true;
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

OperatorConf Get1DZeroVariableOpConf(std::string name, OpConfMap* inserted_ops) {
  OperatorConf variable_op_conf{};
  variable_op_conf.set_name(name);
  VariableOpConf* variable_conf = variable_op_conf.mutable_variable_conf();
  variable_conf->set_out("out");
  *variable_conf->mutable_shape()->mutable_dim()->Add() = 1;
  variable_conf->set_data_type(DataType::kFloat);
  variable_conf->mutable_split_axis()->clear_value();
  variable_conf->mutable_initializer()->mutable_constant_conf()->set_value(0);
  (*inserted_ops)[name] = variable_op_conf;
  return variable_op_conf;
}

OpNode* GetInferenceOutputNode(const OpGraph& op_graph, OpNode* node) {
  OpNode* cur_node = node;
  if (node->op().op_conf().user_conf().op_type_name() == "conv2d") {
    assert(node->out_edges().size() == 1);
    OpNode* next_node = node->SoleOutEdge()->dst_node();
    if (OpTypeName4OpNode(next_node) == "bias_add") {
      cur_node = next_node;
      next_node = next_node->SoleOutEdge()->dst_node();
    }
    if (OpTypeName4OpNode(next_node) == "normalization") {
      cur_node = next_node;
      next_node = next_node->SoleOutEdge()->dst_node();
    }
    if (OpTypeName4OpNode(next_node) == "relu") { cur_node = next_node; }
  }
  VLOG(3) << "For node: " << node->op().op_name();
  VLOG(3) << "output node is: " << cur_node->op().op_name();
  return cur_node;
}

user_op::UserOpConfWrapper MultiplyOp(const std::string& name, const std::string& x,
                                      const std::string& y, OpConfMap* inserted_ops) {
  const auto op_wrapper = user_op::UserOpConfWrapperBuilder(name)
                              .Op("multiply")
                              .Input("x", x)
                              .Input("y", y)
                              .Output("out")
                              .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

user_op::UserOpConfWrapper MinMaxObserver(const std::string& name, const std::string& input,
                                          bool symmetric, bool per_channel,
                                          OpConfMap* inserted_ops) {
  const auto op_wrapper =
      user_op::UserOpConfWrapperBuilder(name)
          .Op("min_max_observer")
          .Input("in", input)
          .Output("scale")
          .Output("zero_point")
          .Attr<std::string>("quantize_scheme", symmetric ? "symmetric" : "affine")
          .Attr("per_layer_quantize", !per_channel)
          .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

user_op::UserOpConfWrapper MovingMinMaxObserver(const std::string& name, const std::string& input,
                                                const std::string& train_step_lbn, bool symmetric,
                                                int64_t stop_update_after_iters, float momentum,
                                                OpConfMap* inserted_ops) {
  const std::string moving_max_name = name + moving_max_suffix;
  const std::string moving_min_name = name + moving_min_suffix;
  const auto moving_max_var = Get1DZeroVariableOpConf(moving_max_name, inserted_ops);
  const auto moving_min_var = Get1DZeroVariableOpConf(moving_min_name, inserted_ops);
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
          .Attr("stop_update_after_iters", stop_update_after_iters)
          .Attr<std::string>("quantize_scheme", symmetric ? "symmetric" : "affine")
          .Attr("momentum", momentum)
          .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

user_op::UserOpConfWrapper FakeQuantOp(const std::string& name, const std::string& input,
                                       const std::string& scale, const std::string& zero_point,
                                       OpConfMap* inserted_ops) {
  const auto op_wrapper = user_op::UserOpConfWrapperBuilder(name)
                              .Op("fake_quantization")
                              .Input("in", input)
                              .Input("scale", scale)
                              .Input("zero_point", zero_point)
                              // affine is always correct
                              .Output("out")
                              .Build();
  (*inserted_ops)[name] = op_wrapper.op_conf();
  return op_wrapper;
}

void GetScaleAndZeroPointLbn4Edge(OpEdge* edge, const std::string train_step_lbn,
                                  std::string* scale, std::string* zero_point,
                                  const QatConfig& qat_config, OpConfMap* inserted_ops) {
  std::string lbn = GenLogicalBlobName(edge->lbis().front());
  std::string conv_input_scale_lbn;
  std::string conv_weight_scale_lbn;
  if (IsConvBiasEdge(edge, &conv_input_scale_lbn, &conv_weight_scale_lbn)) {
    // mul scale
    const std::string mul_scale_op_name = ReplaceSlashToDash4Lbn(lbn) + mul_bias_suffix;
    assert(inserted_ops->find(mul_scale_op_name) == inserted_ops->end());
    const auto mul_scale_op =
        MultiplyOp(mul_scale_op_name, conv_input_scale_lbn, conv_weight_scale_lbn, inserted_ops);

    *scale = mul_scale_op.output("out", 0);
    const std::string zp_var_name = ReplaceSlashToDash4Lbn(lbn) + zp_suffix;
    const auto zp_var = Get1DZeroVariableOpConf(zp_var_name, inserted_ops);
    *zero_point = GenLogicalBlobName(zp_var.name(), zp_var.variable_conf().out());
  } else {
    const std::string observer_op_name = ReplaceSlashToDash4Lbn(lbn) + observer_suffix;
    if (IsWeightEdge(edge)) {
      const auto observer_op =
          MinMaxObserver(observer_op_name, lbn, qat_config.symmetric(),
                         qat_config.per_channel_weight_quantization(), inserted_ops);
      *scale = observer_op.output("scale", 0);
      *zero_point = observer_op.output("zero_point", 0);
    } else {
      assert(qat_config.has_moving_min_max_stop_update_after_iters());
      const auto observer_op =
          MovingMinMaxObserver(observer_op_name, lbn, train_step_lbn, qat_config.symmetric(),
                               qat_config.moving_min_max_stop_update_after_iters(),
                               qat_config.moving_min_max_momentum(), inserted_ops);
      *scale = observer_op.output("scale", 0);
      *zero_point = observer_op.output("zero_point", 0);
    }
  }
}

void ReplaceInputLbn4DstNodeOfEdge(OpEdge* edge, const std::string& new_lbn,
                                   OpConfCache* op_conf_cache) {
  OpNode* dst_node = edge->dst_node();
  LogicalBlobId cur_lbi = edge->lbis().front();
  CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
  const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

  OperatorConf dst_op_conf = op_conf_cache->GetLatest(dst_node->op().op_conf());
  ReplaceInputLbnInOpCustomizedConf(&dst_op_conf, dst_ibn, new_lbn);
  op_conf_cache->Put(dst_op_conf);
}

class QuantAwareTraining final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(QuantAwareTraining);
  QuantAwareTraining() : int8_list_(Int8List()), transparent_list_(TransparentList()) {}
  ~QuantAwareTraining() = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().job_conf().enable_quantization_aware_training();
  }

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;

 private:
  void InsertFakeQuantOp(const QatConfig& qat_config, const OpGraph& op_graph,
                         const QATList& int8_list, HashSet<OpNode*> downstream_white,
                         Job* job) const;

  const QATList& int8_list_;
  const QATList& transparent_list_;
};

Maybe<void> QuantAwareTraining::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  CHECK(GlobalJobDesc().DefaultDataType() == DataType::kFloat);

  VerifyQATList(int8_list_);
  VerifyQATList(transparent_list_);

  std::function<std::string(OpNode* const&)> OpName4Node = [](OpNode* const& node) {
    return node->op().op_name();
  };
  HashSet<OpNode*> white_set;
  DfsTopoGraphTraversal(op_graph, false,
                        [this](OpNode* node) { return IsNodeInList(int8_list_, node); },
                        [&](OpNode* node) { return IsNodeInList(transparent_list_, node); },
                        [&](OpNode* node) { return IsKeyFound(white_set, node); },
                        [&](OpNode* node) {
                          INSERT_CHECK(white_set.insert(node));
                          if (node->op().op_conf().user_conf().op_type_name() == "conv2d") {
                            assert(node->out_edges().size() == 1);
                            OpNode* next_node = node->SoleOutEdge()->dst_node();
                            if (OpTypeName4OpNode(next_node) == "bias_add") {
                              INSERT_CHECK(white_set.insert(next_node));
                              // TODO: mark these special nodes
                              next_node = next_node->SoleOutEdge()->dst_node();
                            }
                            if (OpTypeName4OpNode(next_node) == "normalization") {
                              INSERT_CHECK(white_set.insert(next_node));
                              next_node = next_node->SoleOutEdge()->dst_node();
                            }
                            if (OpTypeName4OpNode(next_node) == "relu") {
                              INSERT_CHECK(white_set.insert(next_node));
                            }
                          }
                        });

  VLOG(3) << "white_set include: "
          << Container2Str<HashSet<OpNode*>, OpNode*>(white_set, OpName4Node);

  InsertFakeQuantOp(ctx->job_desc().job_conf().qat_config(), op_graph, int8_list_, white_set, job);
  return Maybe<void>::Ok();
}

void QuantAwareTraining::InsertFakeQuantOp(const QatConfig& qat_config, const OpGraph& op_graph,
                                           const QATList& int8_list, HashSet<OpNode*> white_set,
                                           Job* job) const {
  JobBuilder job_builder(job);
  HashSet<OpEdge*> white_set_edges;
  auto EdgeName4Edge = [](OpEdge* const& edge) {
    return std::string("edge of\t") + edge->src_node()->op().op_name() + "\tto\t"
           + edge->dst_node()->op().op_name();
  };
  auto AddWhiteSetEdge = [&white_set_edges, &EdgeName4Edge](OpEdge* edge) {
    VLOG(3) << "insert " << EdgeName4Edge(edge);
    assert(edge->lbis().size() == 1);
    const std::string lbn = GenLogicalBlobName(edge->lbis().front());
    scale_map[lbn] = ReplaceSlashToDash4Lbn(lbn) + observer_suffix + "/scale_0";
    VLOG(3) << "set " << lbn << " to " << scale_map[lbn];
    INSERT_CHECK(white_set_edges.insert(edge));
  };
  auto PropagateScale = [](OpNode* node) {
    assert(node->in_edges().size() == 1);
    assert(node->SoleInEdge()->lbis().size() == 1);
    for (OpEdge* edge : node->out_edges()) {
      assert(edge->lbis().size() == 1);
      const std::string node_input_lbn = GenLogicalBlobName(node->SoleInEdge()->lbis().front());
      const std::string lbn = GenLogicalBlobName(edge->lbis().front());
      if (scale_map.find(node_input_lbn) != scale_map.end()) {
        scale_map[lbn] = scale_map[node_input_lbn];
      }
    }
  };

  {
    op_graph.ForEachNode([&](OpNode* node) {
      if (IsKeyFound(white_set, node)) {
        for (OpEdge* edge : node->in_edges()) {
          if (!IsKeyFound(white_set, edge->src_node())) { AddWhiteSetEdge(edge); }
        }
        if (IsNodeInList(int8_list, node)) {
          OpNode* inference_node = GetInferenceOutputNode(op_graph, node);
          for (OpEdge* edge : inference_node->out_edges()) { AddWhiteSetEdge(edge); }
        } else if (IsNodeInList(transparent_list_, node)) {
          PropagateScale(node);
        } else {
          // bias_add/relu/bn in conv -> bias_add -> bn -> relu
          // do nothing
        }
      }
    });
    VLOG(3) << "white_set_edges: "
            << Container2Str<HashSet<OpEdge*>, OpEdge*>(white_set_edges, EdgeName4Edge);
  }

  // group edges by lbn so that we can use `src_node` when calling `AddOps`
  HashMap<std::string, std::vector<OpEdge*>> edges_group_by_lbn;
  {
    for (OpEdge* edge : white_set_edges) {
      CHECK_EQ(1, edge->lbis().size());
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
      GetScaleAndZeroPointLbn4Edge(edge, job->job_conf().train_conf().train_step_lbn(), &scale,
                                   &zero_point, qat_config, &inserted_ops);
      const std::string fake_quant_op_name = ReplaceSlashToDash4Lbn(lbn) + fake_quant_suffix;
      const auto fake_quant_op =
          FakeQuantOp(fake_quant_op_name, lbn, scale, zero_point, &inserted_ops);

      const std::string fake_quant_op_output_name = fake_quant_op.output("out", 0);

      ReplaceInputLbn4DstNodeOfEdge(edge, fake_quant_op_output_name, &op_conf_cache);
    }

    for (const auto& pair : inserted_ops) {
      VLOG(3) << "Insert op: " << pair.second.DebugString() << " between " << lbn;
      job_builder.AddOps(src_node->parallel_desc().parallel_conf(), {pair.second});
    }
  }

  job_builder.MutOpsOnlyOnce(op_conf_cache.op_confs());
}

REGISTER_JOB_PASS("QuantAwareTraining", QuantAwareTraining);

}  // namespace

}  // namespace oneflow

#endif  // WITH_CUDA
