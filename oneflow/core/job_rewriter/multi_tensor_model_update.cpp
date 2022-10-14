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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

struct SGDOptimizerKey {
  std::string learning_rate;
  std::string scale_by_tensor_lbn;
  std::string skip_if_lbn;
  double scale;
  float l1;
  float l2;
  float weight_decay;
  ParallelConf parallel_conf;
  bool has_model_copy;
  /*
  In fuse_model_update_cast pass, not all the cast fp16 model_diff kernel can be fused,
  it may cause some model diff type is float16, some is float32.
  So here we need to use model_diff datatype as key to group.
  */
  DataType model_diff_dtype;
};

bool operator==(const SGDOptimizerKey& lhs, const SGDOptimizerKey& rhs) {
  return (lhs.learning_rate == rhs.learning_rate)
         && (lhs.scale_by_tensor_lbn == rhs.scale_by_tensor_lbn)
         && (lhs.skip_if_lbn == rhs.skip_if_lbn) && (lhs.scale == rhs.scale) && (lhs.l1 == rhs.l1)
         && (lhs.l2 == rhs.l2) && (lhs.weight_decay == rhs.weight_decay)
         && (lhs.parallel_conf == rhs.parallel_conf) && (lhs.has_model_copy == rhs.has_model_copy)
         && (lhs.model_diff_dtype == rhs.model_diff_dtype);
}

struct AdamOptimizerKey {
  std::string learning_rate;
  std::string scale_by_tensor_lbn;
  std::string skip_if_lbn;
  double scale;
  float l1;
  float l2;
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
  bool amsgrad;
  bool do_bias_correction;
  ParallelConf parallel_conf;
  bool has_model_copy;
  DataType model_diff_dtype;
};

bool operator==(const AdamOptimizerKey& lhs, const AdamOptimizerKey& rhs) {
  return (lhs.learning_rate == rhs.learning_rate)
         && (lhs.scale_by_tensor_lbn == rhs.scale_by_tensor_lbn)
         && (lhs.skip_if_lbn == rhs.skip_if_lbn) && (lhs.scale == rhs.scale) && (lhs.l1 == rhs.l1)
         && (lhs.l2 == rhs.l2) && (lhs.beta1 == rhs.beta1) && (lhs.beta2 == rhs.beta2)
         && (lhs.epsilon == rhs.epsilon) && (lhs.weight_decay == rhs.weight_decay)
         && (lhs.amsgrad == rhs.amsgrad) && (lhs.do_bias_correction == rhs.do_bias_correction)
         && (lhs.parallel_conf == rhs.parallel_conf) && (lhs.has_model_copy == rhs.has_model_copy)
         && (lhs.model_diff_dtype == rhs.model_diff_dtype);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::SGDOptimizerKey> {
  size_t operator()(const oneflow::SGDOptimizerKey& key) const {
    const auto float_hash = std::hash<float>();
    const auto double_hash = std::hash<float>();
    const auto& string_hash = std::hash<std::string>();
    const auto& parallel_conf_hash = std::hash<oneflow::ParallelConf>();
    const auto& bool_hash = std::hash<bool>();
    const auto& dtype_hash = std::hash<oneflow::DataType>();

    size_t hash = string_hash(key.learning_rate);
    oneflow::HashCombine(&hash, string_hash(key.scale_by_tensor_lbn));
    oneflow::HashCombine(&hash, string_hash(key.skip_if_lbn));
    oneflow::HashCombine(&hash, double_hash(key.scale));
    oneflow::HashCombine(&hash, float_hash(key.l1));
    oneflow::HashCombine(&hash, float_hash(key.l2));
    oneflow::HashCombine(&hash, float_hash(key.weight_decay));
    oneflow::HashCombine(&hash, parallel_conf_hash(key.parallel_conf));
    oneflow::HashCombine(&hash, bool_hash(key.has_model_copy));
    oneflow::HashCombine(&hash, dtype_hash(key.model_diff_dtype));
    return hash;
  }
};

template<>
struct hash<oneflow::AdamOptimizerKey> {
  size_t operator()(const oneflow::AdamOptimizerKey& key) const {
    const auto& float_hash = std::hash<float>();
    const auto& double_hash = std::hash<float>();
    const auto& string_hash = std::hash<std::string>();
    const auto& bool_hash = std::hash<bool>();
    const auto& parallel_conf_hash = std::hash<oneflow::ParallelConf>();
    const auto& dtype_hash = std::hash<oneflow::DataType>();

    size_t hash = string_hash(key.learning_rate);
    oneflow::HashCombine(&hash, string_hash(key.scale_by_tensor_lbn));
    oneflow::HashCombine(&hash, string_hash(key.skip_if_lbn));
    oneflow::HashCombine(&hash, double_hash(key.scale));
    oneflow::HashCombine(&hash, float_hash(key.l1));
    oneflow::HashCombine(&hash, float_hash(key.l2));
    oneflow::HashCombine(&hash, float_hash(key.beta1));
    oneflow::HashCombine(&hash, float_hash(key.beta2));
    oneflow::HashCombine(&hash, float_hash(key.epsilon));
    oneflow::HashCombine(&hash, float_hash(key.weight_decay));
    oneflow::HashCombine(&hash, bool_hash(key.amsgrad));
    oneflow::HashCombine(&hash, bool_hash(key.do_bias_correction));
    oneflow::HashCombine(&hash, parallel_conf_hash(key.parallel_conf));
    oneflow::HashCombine(&hash, bool_hash(key.has_model_copy));
    oneflow::HashCombine(&hash, dtype_hash(key.model_diff_dtype));
    return hash;
  }
};

}  // namespace std

namespace oneflow {

namespace {

void AddScaleAndSkipLbn(user_op::UserOpConfWrapperBuilder& multi_tensor_model_update_op_builder,
                        const user_op::UserOpConfWrapper& model_update_user_conf) {
  if (model_update_user_conf.has_input("scale_by_tensor", 0)) {
    multi_tensor_model_update_op_builder.Input("scale_by_tensor",
                                               model_update_user_conf.input("scale_by_tensor", 0));
  }
  if (model_update_user_conf.has_input("skip_if", 0)) {
    multi_tensor_model_update_op_builder.Input("skip_if",
                                               model_update_user_conf.input("skip_if", 0));
  }
}

void AddProcessedVariable(HashSet<std::string>& processed_variable_list,
                          const user_op::UserOpConfWrapper& model_update_user_conf) {
  /*
  Since each variable op will be processed in pass, for example, Adam optimizer has 3 variables:
  model, m, v. We replace to multi tensor optimizer and processed 3 variables at once, if we don't
  filter these variables, these variables will be repeated 3 times in multi_tensor_update kernel.

  Here we use a HashSet to sign if the variable has been processed.
  */
  processed_variable_list.emplace(model_update_user_conf.input("model", 0));
  if (model_update_user_conf.op_type_name() == "adam_update") {
    processed_variable_list.emplace(model_update_user_conf.input("m", 0));
    processed_variable_list.emplace(model_update_user_conf.input("v", 0));
  }
}

bool IfVariableProcessed(const HashSet<std::string>& processed_variable_list,
                         const user_op::UserOpConfWrapper& model_update_user_conf) {
  if (model_update_user_conf.op_type_name() == "sgd_update") {
    const auto& processed_model_iter =
        processed_variable_list.find(model_update_user_conf.input("model", 0));
    if (processed_model_iter != processed_variable_list.end()) { return true; }
  } else if (model_update_user_conf.op_type_name() == "adam_update") {
    const auto& processed_model_iter =
        processed_variable_list.find(model_update_user_conf.input("model", 0));
    const auto& processed_m_iter =
        processed_variable_list.find(model_update_user_conf.input("m", 0));
    const auto& processed_v_iter =
        processed_variable_list.find(model_update_user_conf.input("v", 0));
    if (processed_model_iter != processed_variable_list.end()
        && processed_m_iter != processed_variable_list.end()
        && processed_v_iter != processed_variable_list.end()) {
      return true;
    }
  } else {
    UNIMPLEMENTED() << "Current Optimizer do not support multi tensor update. ";
  }
  return false;
}

class MultiTensorModelUpdatePass final : public JobPass {
 public:
  MultiTensorModelUpdatePass() = default;
  ~MultiTensorModelUpdatePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().enable_multi_tensor_update()
           || ParseBooleanFromEnv("ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE", false);
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> MultiTensorModelUpdatePass::Apply(const OpGraph& op_graph,
                                              JobBuilder* job_builder) const {
  if (!job_builder->job().job_conf().has_train_conf()) { return Maybe<void>::Ok(); }
  std::vector<OperatorConf> delete_ops;
  ParallelConf parallel_conf{};
  HashMap<SGDOptimizerKey, user_op::UserOpConfWrapperBuilder> multi_tensor_sgd_update_hashmap;
  HashMap<AdamOptimizerKey, user_op::UserOpConfWrapperBuilder> multi_tensor_adam_update_hashmap;
  HashSet<std::string> processed_variable_list{};

  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return; }
    LogicalBlobId model_copy_lbi;

    for (OpEdge* find_model_update_edge : op_node->out_edges()) {
      OpNode* find_model_update_update_node = find_model_update_edge->dst_node();
      if (!IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(), "sgd_update")
          && !IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(), "adam_update")) {
        continue;
      }
      const user_op::UserOpConfWrapper model_update_user_conf(
          find_model_update_update_node->op().op_conf());
      // Multi tensor update pass only support for CUDA currently.
      if (find_model_update_update_node->parallel_desc().device_type() != DeviceType::kCUDA) {
        continue;
      }

      // Multi tensor update pass only support Data Parallel.
      bool if_data_parallel = true;
      for (const auto& pair :
           find_model_update_update_node->sbp_signature().bn_in_op2sbp_parallel()) {
        if (!pair.second.has_broadcast_parallel()) {
          if_data_parallel = false;
          break;
        }
      }
      if (!if_data_parallel) { continue; }

      // Check the variable has been processed before.
      if (IfVariableProcessed(processed_variable_list, model_update_user_conf)) { continue; }

      delete_ops.emplace_back(find_model_update_update_node->op().op_conf());
      parallel_conf = find_model_update_update_node->parallel_desc().parallel_conf();

      std::string scale_by_tensor_lbn = "";
      std::string skip_if_lbn = "";
      bool has_model_copy = false;
      if (model_update_user_conf.has_input("scale_by_tensor", 0)) {
        scale_by_tensor_lbn = model_update_user_conf.input("scale_by_tensor", 0);
      }
      if (model_update_user_conf.has_input("skip_if", 0)) {
        skip_if_lbn = model_update_user_conf.input("skip_if", 0);
      }
      if (model_update_user_conf.has_input("model_copy", 0)) { has_model_copy = true; }

      const BlobDesc& model_diff_blob_desc = op_graph.GetLogicalBlobDesc(
          GenLogicalBlobId(model_update_user_conf.input("model_diff", 0)));
      const DataType model_diff_dtype = model_diff_blob_desc.data_type();

      if (IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(), "sgd_update")) {
        SGDOptimizerKey key{model_update_user_conf.input("learning_rate", 0),
                            scale_by_tensor_lbn,
                            skip_if_lbn,
                            model_update_user_conf.attr<double>("scale"),
                            model_update_user_conf.attr<float>("l1"),
                            model_update_user_conf.attr<float>("l2"),
                            model_update_user_conf.attr<float>("weight_decay"),
                            parallel_conf,
                            has_model_copy,
                            model_diff_dtype};
        const auto& iter = multi_tensor_sgd_update_hashmap.find(key);

        if (iter != multi_tensor_sgd_update_hashmap.end()) {
          iter->second.Input("model", model_update_user_conf.input("model", 0))
              .Input("model_diff", model_update_user_conf.input("model_diff", 0));
          if (has_model_copy) {
            iter->second.Input("model_copy", model_update_user_conf.input("model_copy", 0));
          }
        } else {
          user_op::UserOpConfWrapperBuilder multi_tensor_sgd_update_op_builder(
              "multi_tensor_model_update" + NewUniqueId());
          std::string op_type_name = "multi_tensor_sgd_update";
          if (has_model_copy) { op_type_name = "multi_tensor_sgd_update_with_cast"; }

          multi_tensor_sgd_update_op_builder.OpTypeName(op_type_name)
              .Input("model", model_update_user_conf.input("model", 0))
              .Input("model_diff", model_update_user_conf.input("model_diff", 0))
              .Input("learning_rate", model_update_user_conf.input("learning_rate", 0))
              .Attr<double>("scale", model_update_user_conf.attr<double>("scale"))
              .Attr<float>("l1", model_update_user_conf.attr<float>("l1"))
              .Attr<float>("l2", model_update_user_conf.attr<float>("l2"))
              .Attr<float>("weight_decay", model_update_user_conf.attr<float>("weight_decay"));
          if (has_model_copy) {
            multi_tensor_sgd_update_op_builder.Input("model_copy",
                                                     model_update_user_conf.input("model_copy", 0));
          }

          AddScaleAndSkipLbn(multi_tensor_sgd_update_op_builder, model_update_user_conf);

          CHECK(model_update_user_conf.op_conf().has_scope_symbol_id());
          multi_tensor_sgd_update_op_builder.ScopeSymbolId(
              model_update_user_conf.op_conf().scope_symbol_id());
          multi_tensor_sgd_update_hashmap.emplace(key, multi_tensor_sgd_update_op_builder);
        }
      } else if (IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(),
                                      "adam_update")) {
        AdamOptimizerKey key{model_update_user_conf.input("learning_rate", 0),
                             scale_by_tensor_lbn,
                             skip_if_lbn,
                             model_update_user_conf.attr<double>("scale"),
                             model_update_user_conf.attr<float>("l1"),
                             model_update_user_conf.attr<float>("l2"),
                             model_update_user_conf.attr<float>("beta1"),
                             model_update_user_conf.attr<float>("beta2"),
                             model_update_user_conf.attr<float>("epsilon"),
                             model_update_user_conf.attr<float>("weight_decay"),
                             model_update_user_conf.attr<bool>("amsgrad"),
                             model_update_user_conf.attr<bool>("do_bias_correction"),
                             parallel_conf,
                             has_model_copy,
                             model_diff_dtype};
        if (key.amsgrad) {
          UNIMPLEMENTED() << "Multi Tensor Adam update do not support amsgrad = True. ";
        }
        const auto& iter = multi_tensor_adam_update_hashmap.find(key);

        if (iter != multi_tensor_adam_update_hashmap.end()) {
          iter->second.Input("model", model_update_user_conf.input("model", 0))
              .Input("model_diff", model_update_user_conf.input("model_diff", 0))
              .Input("m", model_update_user_conf.input("m", 0))
              .Input("v", model_update_user_conf.input("v", 0));
          if (has_model_copy) {
            iter->second.Input("model_copy", model_update_user_conf.input("model_copy", 0));
          }
          if (model_update_user_conf.attr<bool>("do_bias_correction")) {
            iter->second
                .Input("bias_correction1", model_update_user_conf.input("bias_correction1", 0))
                .Input("bias_correction2", model_update_user_conf.input("bias_correction2", 0));
          }
        } else {
          user_op::UserOpConfWrapperBuilder multi_tensor_adam_update_op_builder(
              "multi_tensor_model_update" + NewUniqueId());
          std::string op_type_name = "multi_tensor_adam_update";
          if (has_model_copy) { op_type_name = "multi_tensor_adam_update_with_cast"; }
          multi_tensor_adam_update_op_builder.OpTypeName(op_type_name)
              .Input("model", model_update_user_conf.input("model", 0))
              .Input("model_diff", model_update_user_conf.input("model_diff", 0))
              .Input("m", model_update_user_conf.input("m", 0))
              .Input("v", model_update_user_conf.input("v", 0))
              .Input("learning_rate", model_update_user_conf.input("learning_rate", 0))
              .Attr<double>("scale", model_update_user_conf.attr<double>("scale"))
              .Attr<float>("l1", model_update_user_conf.attr<float>("l1"))
              .Attr<float>("l2", model_update_user_conf.attr<float>("l2"))
              .Attr<float>("beta1", model_update_user_conf.attr<float>("beta1"))
              .Attr<float>("beta2", model_update_user_conf.attr<float>("beta2"))
              .Attr<float>("epsilon", model_update_user_conf.attr<float>("epsilon"))
              .Attr<float>("weight_decay", model_update_user_conf.attr<float>("weight_decay"))
              .Attr<bool>("amsgrad", model_update_user_conf.attr<bool>("amsgrad"))
              .Attr<bool>("do_bias_correction",
                          model_update_user_conf.attr<bool>("do_bias_correction"));

          if (model_update_user_conf.attr<bool>("do_bias_correction")) {
            multi_tensor_adam_update_op_builder
                .Input("bias_correction1", model_update_user_conf.input("bias_correction1", 0))
                .Input("bias_correction2", model_update_user_conf.input("bias_correction2", 0));
          }
          if (has_model_copy) {
            multi_tensor_adam_update_op_builder.Input(
                "model_copy", model_update_user_conf.input("model_copy", 0));
          }
          AddScaleAndSkipLbn(multi_tensor_adam_update_op_builder, model_update_user_conf);

          CHECK(model_update_user_conf.op_conf().has_scope_symbol_id());
          multi_tensor_adam_update_op_builder.ScopeSymbolId(
              model_update_user_conf.op_conf().scope_symbol_id());
          multi_tensor_adam_update_hashmap.emplace(key, multi_tensor_adam_update_op_builder);
        }
      } else {
        UNIMPLEMENTED() << "Current Optimizer do not support multi tensor update. ";
      }

      AddProcessedVariable(processed_variable_list, model_update_user_conf);
      break;
    }
  });
  for (auto& op : multi_tensor_sgd_update_hashmap) {
    auto multi_tensor_model_update_sgd_op = op.second.Build();
    job_builder->AddOps(parallel_conf, {multi_tensor_model_update_sgd_op.op_conf()});
  }
  for (auto& op : multi_tensor_adam_update_hashmap) {
    auto multi_tensor_model_update_adam_op = op.second.Build();
    job_builder->AddOps(parallel_conf, {multi_tensor_model_update_adam_op.op_conf()});
  }
  job_builder->DelOps(delete_ops);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("MultiTensorModelUpdatePass", MultiTensorModelUpdatePass);

}  // namespace oneflow
