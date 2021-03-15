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
#include "oneflow/core/framework/op_interpreter_util.h"

#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {
namespace one {

// Our system will has only 4 kind interpreters.
enum class OpInterpKind : int {
  kLazyConsistent = 0,
  kLazyMirrored = 1,
  kEagerConsistent = 2,
  kEagerMirrored = 3,

  // Interpreter kind size.
  kOpInterpKindSize = 4
};

static std::shared_ptr<OpExprInterpreter> BuildInterpreter(const bool& eager_mode,
                                                           const bool& mirrored_mode) {
  std::shared_ptr<OpExprInterpContext> context(
      new OpExprInterpContext{.is_mirrored_strategy_enabled = mirrored_mode});
  std::shared_ptr<NormalInterpreter> normal_interp;
  if (eager_mode) {
    normal_interp = std::make_shared<EagerInterpreter>(context);
  } else {
    normal_interp = std::make_shared<LazyInterpreter>(context);
  }
  return std::make_shared<AutogradInterpreter>(normal_interp);
}

/*static*/ Maybe<OpExprInterpreter> OpInterpUtil::GetInterpreter() {
  thread_local static std::vector<std::shared_ptr<OpExprInterpreter>> all_interpreters(
      static_cast<int>(OpInterpKind::kOpInterpKindSize));
  const auto& session = JUST(GetDefaultSession());
  int mirrored_mode = session->IsMirroredStrategyEnabled();
  int eager_mode = EagerExecutionEnabled();
  int kind = (eager_mode >> 1) + mirrored_mode;
  CHECK_LT_OR_RETURN(kind, static_cast<int>(OpInterpKind::kOpInterpKindSize));
  if (!all_interpreters[kind].get()) {
    all_interpreters[kind] = BuildInterpreter(eager_mode, mirrored_mode);
  }
  return all_interpreters[kind];
}

/*static*/ Maybe<cfg::OpAttribute> OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
    const OperatorConf& op_conf, const bool is_mirrored_strategy_enabled) {
  std::shared_ptr<OpAttribute> op_attribute = JUST([&]() -> Maybe<OpAttribute> {
    auto infer_ctx = JUST(GetCurInferCtx());
    if (is_mirrored_strategy_enabled) {
      return infer_ctx->AddAndInferMirroredOp(op_conf);
    } else {
      return infer_ctx->AddAndInferConsistentOp(op_conf);
    }
  }());
  return std::make_shared<cfg::OpAttribute>(*op_attribute);
}

/*static*/ Maybe<cfg::OpAttribute> OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
    const BuiltinOpExpr* op_expr, const std::shared_ptr<Scope>& scope,
    const bool is_mirrored_strategy_enabled) {
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr));
  int64_t symbol_id = JUST(scope->symbol_id());
  op_conf->set_scope_symbol_id(symbol_id);
  if (!op_conf->has_device_tag()) {
    op_conf->set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  return OpInterpUtil::AddBuiltinOpAndInferOpAttribute(*op_conf, is_mirrored_strategy_enabled);
}

using Bn2BlobObjectMap = HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>;

/*static*/ Maybe<Bn2BlobObjectMap> OpInterpUtil::MakeBn2BlobObjectMap(
    const std::vector<std::string>& indexed_ibns, const TensorTuple& inputs) {
  CHECK_EQ_OR_RETURN(indexed_ibns.size(), inputs.size());
  auto* bn2blob_object(new HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>{});
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& ibn = indexed_ibns.at(i);
    const auto& blob_object = JUST(OpInterpUtil::GetTensorBlobObject(inputs[i]));
    bn2blob_object->emplace(ibn, blob_object);
  }
  return std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>(
      bn2blob_object);
}

/*static*/ Maybe<OperatorConf> OpInterpUtil::GenBuiltinOpConf(const BuiltinOpExpr* op_expr) {
  auto* op_conf = new OperatorConf;
  op_expr->BuildOpConf(op_conf);
  return std::shared_ptr<OperatorConf>(op_conf);
}

/*static*/ Maybe<OperatorConf> OpInterpUtil::GenModelInitOpConf(const OperatorConf& variable_conf) {
  auto* model_init_op_conf = new OperatorConf;
  model_init_op_conf->set_name("model_init");
  model_init_op_conf->set_device_tag("cpu");
  model_init_op_conf->mutable_model_init_conf()->mutable_out()->Add()->assign("out_0");
  model_init_op_conf->mutable_model_init_conf()->mutable_variable_op_name()->Add()->assign(
      variable_conf.name());
  model_init_op_conf->mutable_model_init_conf()->mutable_original_variable_conf()->Add()->CopyFrom(
      variable_conf);
  return std::shared_ptr<OperatorConf>(model_init_op_conf);
}

/*static*/ Maybe<OperatorConf> OpInterpUtil::GenModelIOPathInputOpConf() {
  auto* path_input_op_conf = new OperatorConf;
  path_input_op_conf->set_name("model_io_path_input");
  path_input_op_conf->set_device_tag("cpu");
  path_input_op_conf->mutable_input_conf()->set_out("out");

  InterfaceBlobConf blob_conf;
  blob_conf.mutable_shape()->mutable_dim()->Add(65536);
  blob_conf.set_data_type(kInt8);
  blob_conf.set_is_dynamic(true);

  path_input_op_conf->mutable_input_conf()->mutable_blob_conf()->CopyFrom(blob_conf);
  return std::shared_ptr<OperatorConf>(path_input_op_conf);
}

/*static*/ Maybe<OperatorConf> OpInterpUtil::GenModelLoadOpConf(
    const OperatorConf& variable_conf, const OperatorConf& path_input_op_conf) {
  auto* model_load_op_conf = new OperatorConf;
  model_load_op_conf->set_name("model_load");
  model_load_op_conf->set_device_tag("cpu");

  CHECK(path_input_op_conf.has_model_init_conf());
  std::string path =
      path_input_op_conf.name() + "/" + path_input_op_conf.model_init_conf().out()[0];
  model_load_op_conf->mutable_model_load_conf()->set_path(path);
  model_load_op_conf->mutable_model_load_conf()->mutable_out()->Add()->assign("out_0");
  model_load_op_conf->mutable_model_load_conf()->mutable_variable_op_name()->Add()->assign(
      variable_conf.name());
  model_load_op_conf->mutable_model_load_conf()->mutable_original_variable_conf()->Add()->CopyFrom(
      variable_conf);
  return std::shared_ptr<OperatorConf>(model_load_op_conf);
}

/*static*/ Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
OpInterpUtil::BuildModelInitOrIOPathInputInstruction(
    const OperatorConf& op_conf, const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object) {
  using namespace std::placeholders;
  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& scope = CHECK_JUST(GetCurrentScope());
    const auto& op_attribute =
        CHECK_JUST(OpInterpUtil::InferOpAttribute(op_conf, scope, Bn2BlobObjectMap{}));
    auto parallel_conf =
        std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
    const auto& boxing_util = (*Global<std::shared_ptr<ForeignBoxingUtil>>::Get());
    CHECK_JUST(builder->StatelessCall(
        op_attribute, parallel_conf, bn2blob_object,
        std::bind(&ForeignBoxingUtil::BoxingTo, boxing_util.get(), _1, _2, _3)));
  };
  return std::function<void(const std::shared_ptr<InstructionsBuilder>&)>(build_instruction);
}

/*static*/ Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
OpInterpUtil::BuildFeedPathInstruction(const std::string& path,
                                       const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object) {
  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    int64_t callback_id = (*Global<std::shared_ptr<ForeignCallback>>::Get())->FeedPath(path);
    const auto& blob_object = bn2blob_object->at("out");
    CHECK_JUST(builder->FeedBlob(blob_object, callback_id));
    CHECK_JUST(
        builder->InsertRemoveForeignCallbackInstruction(blob_object->object_id(), callback_id));
  };
  return std::function<void(const std::shared_ptr<InstructionsBuilder>&)>(build_instruction);
}

/*static*/ Maybe<compatible_py::BlobObject> OpInterpUtil::EagerRunModelInit(
    const OperatorConf& op_conf) {
  auto model_init_conf = JUST(GenModelInitOpConf(op_conf));
  std::shared_ptr<Bn2BlobObjectMap> bn2blob_object(new Bn2BlobObjectMap{});

  auto build_model_init_instruction =
      JUST(BuildModelInitOrIOPathInputInstruction(*model_init_conf, bn2blob_object));
  JUST(LogicalRun(*build_model_init_instruction));
  return bn2blob_object->at("out_0");
}

/*static*/ Maybe<compatible_py::BlobObject> OpInterpUtil::EagerRunModelLoad(
    const OperatorConf& op_conf, const std::string& snapshot_path) {
  using namespace std::placeholders;
  CHECK_OR_RETURN(Basename(snapshot_path) == "out");
  CHECK_OR_RETURN(Dirname(snapshot_path) == op_conf.name());

  const auto& path_input_op_conf = JUST(GenModelIOPathInputOpConf());

  std::shared_ptr<Bn2BlobObjectMap> bn2blob_object(new Bn2BlobObjectMap{});
  auto build_model_io_path_input_instruction =
      JUST(BuildModelInitOrIOPathInputInstruction(*path_input_op_conf, bn2blob_object));
  auto build_feed_path_instruction = JUST(BuildFeedPathInstruction(snapshot_path, bn2blob_object));

  std::shared_ptr<Bn2BlobObjectMap> model_load_blob_objects(new Bn2BlobObjectMap{});
  const auto& model_load_op_conf = JUST(GenModelLoadOpConf(op_conf, *path_input_op_conf));
  auto build_model_load_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& scope = CHECK_JUST(GetCurrentScope());
    const auto& blob_object = bn2blob_object->at("out");
    (*model_load_blob_objects)["path"] = blob_object;
    const auto& op_attribute = CHECK_JUST(
        OpInterpUtil::InferOpAttribute(*model_load_op_conf, scope, *model_load_blob_objects));
    auto parallel_conf =
        std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
    const auto& boxing_util = *Global<std::shared_ptr<ForeignBoxingUtil>>::Get();
    CHECK_JUST(builder->StatelessCall(
        op_attribute, parallel_conf, model_load_blob_objects,
        std::bind(&ForeignBoxingUtil::BoxingTo, boxing_util.get(), _1, _2, _3)));
  };

  JUST(LogicalRun(*build_model_io_path_input_instruction));
  JUST(LogicalRun(*build_feed_path_instruction));
  JUST(LogicalRun(build_model_load_instruction));
  return model_load_blob_objects->at("out_0");
}

/*static*/ Maybe<void> OpInterpUtil::Assign(
    const std::shared_ptr<compatible_py::BlobObject>& target_blob_object,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  auto build_assign_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& boxing_util = *Global<std::shared_ptr<ForeignBoxingUtil>>::Get();
    auto new_parallel_desc_symbol = boxing_util->TryReplaceDeviceTag(
        builder, target_blob_object->parallel_desc_symbol(), "cpu");
    auto consumer_op_arg_parallel_attr = std::make_shared<compatible_py::OpArgParallelAttribute>(
        new_parallel_desc_symbol, target_blob_object->op_arg_parallel_attr()->sbp_parallel(),
        target_blob_object->op_arg_parallel_attr()->opt_mirrored_parallel());
    auto tmp_blob_object =
        boxing_util->BoxingTo(builder, blob_object, consumer_op_arg_parallel_attr);
    boxing_util->Assign(builder, target_blob_object, tmp_blob_object);
  };
  return LogicalRun(build_assign_instruction);
}

/*static*/ Maybe<compatible_py::BlobObject> OpInterpUtil::GetTensorBlobObject(
    const std::shared_ptr<Tensor>& tensor) {
  if (auto* mirrored_tensor = dynamic_cast<MirroredTensor*>(tensor.get())) {
    return mirrored_tensor->blob_object();
  } else if (auto* consistent_tensor = dynamic_cast<ConsistentTensor*>(tensor.get())) {
    return consistent_tensor->blob_object();
  } else {
    CHECK_OR_RETURN(false) << "The tensor should be either Mirrored Tensor or Consistent Tensor.";
  }
}

/*static*/ Maybe<void> OpInterpUtil::InitVariableOutputBlob(const std::shared_ptr<Session>& session,
                                                            const std::shared_ptr<Tensor>& output,
                                                            const OpAttribute& op_attribute) {
  const auto& op_conf = op_attribute.op_conf();
  const auto& snapshot_path = session->snapshot_mgr()->GetSnapshotPath(op_conf.name());

  std::shared_ptr<compatible_py::BlobObject> temp_blob_object;
  if (snapshot_path.empty()) {
    temp_blob_object = JUST(OpInterpUtil::EagerRunModelInit(op_conf));
  } else {
    temp_blob_object = JUST(OpInterpUtil::EagerRunModelLoad(op_conf, snapshot_path));
  }
  auto target_blob_object = JUST(GetTensorBlobObject(output));
  return OpInterpUtil::Assign(target_blob_object, temp_blob_object);
}

/*static*/ Maybe<cfg::OpAttribute> OpInterpUtil::InferOpAttribute(
    const OperatorConf& op_conf, const std::shared_ptr<Scope>& scope,
    const Bn2BlobObjectMap& ibn2blob_object) {
  // TODO(): Remove const_cast.
  auto& mutable_op_conf = const_cast<OperatorConf&>(op_conf);
  const auto& symbol_id = JUST(scope->symbol_id());
  mutable_op_conf.set_scope_symbol_id(symbol_id);
  OpNodeSignature upstream_signature;
  if (ibn2blob_object.size()) {
    std::shared_ptr<cfg::OpNodeSignature> cfg_upstream_signature(new cfg::OpNodeSignature);
    for (const auto& it : ibn2blob_object) {
      it.second->op_arg_parallel_attr()->DumpToOpNodeSignature(it.first, cfg_upstream_signature);
      it.second->op_arg_blob_attr()->DumpToOpNodeSignature(it.first, cfg_upstream_signature);
    }
    cfg_upstream_signature->ToProto(&upstream_signature);
  }
  const auto& op = JUST(ConstructAndInferOp(mutable_op_conf, upstream_signature, *scope));
  const auto& op_attribute = op->GetOpAttributeWithoutOpNameAndLbn();
  return std::make_shared<cfg::OpAttribute>(*op_attribute);
}

/*static*/ Maybe<cfg::OpAttribute> OpInterpUtil::InferOpAttribute(
    const BuiltinOpExpr* op_expr, const std::shared_ptr<Scope>& scope, const TensorTuple& inputs) {
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr));
  int64_t symbol_id = JUST(scope->symbol_id());
  op_conf->set_scope_symbol_id(symbol_id);
  if (!op_conf->has_device_tag()) {
    op_conf->set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  const auto& ibn2blob_object = JUST(MakeBn2BlobObjectMap(op_expr->indexed_ibns(), inputs));
  return OpInterpUtil::InferOpAttribute(*op_conf, scope, *ibn2blob_object);
}

}  // namespace one
}  // namespace oneflow
