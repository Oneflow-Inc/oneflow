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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/kernel/op_kernel.h"
#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/stl.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"

#include <string>
#include <vector>

namespace oneflow {
namespace xrt {

template<typename T>
void DoNoDuplicationAdd(util::PbVector<T> *repeat_field, const T &val) {
  if (std::find(repeat_field->begin(), repeat_field->end(), val) == repeat_field->end()) {
    repeat_field->Add()->assign(val);
  }
}

int GetRepeatedIndex(const std::string &input) {
  auto name_and_index = GetFieldNameAndIndex4StrVal(input);
  return name_and_index.second;
};

void SetOpInputBlobName(OperatorConf *op_conf, const std::string &input,
                        const std::string &blob_name, const std::string &fixed_blob_name) {
  auto *spec_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
  switch (op_conf->op_type_case()) {
    case OperatorConf::kUserConf: {
      std::pair<std::string, int32_t> pair = GetFieldNameAndIndex4StrVal(input);
      auto it = op_conf->user_conf().input().find(pair.first);
      CHECK(it != op_conf->user_conf().input().end());
      CHECK(pair.second >= 0 && pair.second < it->second.s_size());
      CHECK_EQ(it->second.s(pair.second), blob_name);
      (*(op_conf->mutable_user_conf()->mutable_input()))[pair.first].set_s(pair.second,
                                                                           fixed_blob_name);
      break;
    }
    default: {
      const auto& old_val = ReplaceStrValInPbFdOrPbRpf(spec_conf, input, fixed_blob_name);
      CHECK_EQ(old_val, blob_name);
    }
  }
}

class FoldSubgraphBuilder {
 public:
  FoldSubgraphBuilder(const XrtGraph &graph, Job *job);

  virtual ~FoldSubgraphBuilder() {}

  void Build() {
    CHECK(builder_) << "Builder has not been initialized.";
    // Rebuilding folded job should takes the below steps in order.
    // 0.Fixup output blob names for launch nodes, and infect the
    //   changes to the input of next nodes.
    FixupInOutBlobNames();
    // 1.Add XrtLaunch operator to the job.
    BuildXrtLaunchOps();
    // 2.Replace control_in_op_name by the XrtLaunch operator name if
    //   the operator has been folded.
    FixupControlInOpNames();
    // 3.Add time shape for XrtLaunch operators.
    FixupTimeShapes();
    // 4.Add sbp parallel strategy for XrtLaunch operators.
    FixupSbpSignatures();
    // 6.Finally remove the folded operators.
    RemoveLaunchFoldedOps();
  }

 private:
  void buildFunction(const XrtGraph *sub_graph, const XrtEngine &engine,
                     util::Set<std::string> *input_mutability, bool *is_model_update,
                     XrtLaunchOpConf::Function *function) const;

  void BuildXrtLaunchOps();

  void FixupControlInOpNames();

  void FixupTimeShapes();

  void FixupSbpSignatures();

  void FixupInOutBlobNames();

  void RemoveLaunchFoldedOps();

  XrtEngine GraphEngine(const XrtGraph *graph) const;

 private:
  const XrtGraph &graph_;
  std::shared_ptr<JobBuilder> builder_;

  // Launch nodes
  std::vector<const XrtNode *> launch_nodes_;
  // Folded nodes except for argument nodes for each launch nodes
  std::vector<std::vector<const XrtNode *>> folded_nodes_;
  // TODO(hjchen2): Remove this
  util::Set<const XrtNode *> after_allreduce_nodes_;

  util::Map<std::string, std::string> fixedup_names_;
};

FoldSubgraphBuilder::FoldSubgraphBuilder(const XrtGraph &graph, Job *job) : graph_(graph) {
  for (const XrtNode *node : graph_.Nodes()) {
    if (node->type() == _XrtLaunchOpType) { launch_nodes_.push_back(node); }
  }

  folded_nodes_.resize(launch_nodes_.size());
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    XrtGraph *sub_graph = launch_nodes_[i]->sub_graph();
    CHECK_NOTNULL(sub_graph);
    for (const XrtNode *sub_node : sub_graph->Nodes()) {
      if (!sub_node->IsArgumentNode()) { folded_nodes_[i].push_back(sub_node); }
    }
  }
  builder_ = std::make_shared<JobBuilder>(job);
}

bool IsMutableArgument(const Argument &argument, const std::string &op_type,
                       const XrtField &field) {
  const auto &mutable_vars = MutableVariables(op_type, field);
  const std::string &key = argument.meta_data().consume_key;
  return mutable_vars.count(key) > 0;
}

void FoldSubgraphBuilder::buildFunction(const XrtGraph *sub_graph, const XrtEngine &engine,
                                        util::Set<std::string> *input_mutability,
                                        bool *is_model_update,
                                        XrtLaunchOpConf::Function *function) const {
  for (const XrtNode *node : sub_graph->Nodes()) {
    if (!node->IsArgumentNode()) {
      *(function->add_node()) = *reinterpret_cast<const OperatorConf *>(&node->param());
      *is_model_update = IsOptimizerNode(node, engine);
    } else {
      auto *argument_proto = function->add_argument();
      argument_proto->set_name(node->name());
      DeviceType device_type = XrtDeviceToDeviceType(node->device());
      argument_proto->set_device_type(device_type);
      // Usually one argument node has either inputs or outputs
      CHECK(node->in_edges().size() == 0 || node->out_edges().size() == 0);
      bool is_mutable = false;
      // Build inputs or outputs for the argument nodes
      for (const XrtEdge *edge : node->out_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_value(argument.name());

        const std::string &op_type = edge->end()->type();
        const XrtDevice &device = edge->end()->device();
        is_mutable |= IsMutableArgument(argument, op_type, MakeXrtField(device, engine));
      }
      for (const XrtEdge *edge : node->in_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_value(argument.name());
      }
      if (is_mutable) { input_mutability->insert(argument_proto->value()); }
    }
  }
}

void AddInOutBlobNames(const XrtNode *node, XrtLaunchOpConf *launch_conf) {
  // Add inputs
  util::Map<std::string, std::string> input_args;
  for (const XrtEdge *edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      input_args.emplace(arg.meta_data().consume_key, arg.name());
    }
  }
  for (int i = 0; i < input_args.size(); ++i) {
    std::string consume_key = absl::StrCat("in_", i);
    CHECK_GT(input_args.count(consume_key), 0);
    const std::string &val = input_args.at(consume_key);
    launch_conf->mutable_in()->Add()->assign(val);
  }

  // Add outputs
  util::Map<std::string, std::string> output_args;
  for (const XrtEdge *edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      output_args.emplace(arg.meta_data().produce_key, arg.name());
    }
  }
  for (int i = 0; i < output_args.size(); ++i) {
    std::string produce_key = absl::StrCat("out_", i);
    CHECK_GT(output_args.count(produce_key), 0);
    launch_conf->mutable_out()->Add()->assign(produce_key);
  }
}

XrtEngine FoldSubgraphBuilder::GraphEngine(const XrtGraph *graph) const {
  CHECK(graph->HasAttr("engine"));
  return graph->Attr<XrtEngine>("engine");
}

void FoldSubgraphBuilder::BuildXrtLaunchOps() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    const XrtNode *node = launch_nodes_[i];
    // Add xrt launch operator
    OperatorConf op_conf;
    op_conf.set_name(node->name());
    DeviceType device_type = XrtDeviceToDeviceType(node->device());
    op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(device_type)));

    XrtLaunchOpConf *launch_conf = op_conf.mutable_xrt_launch_conf();
    // Add inputs and outputs in launch_conf
    AddInOutBlobNames(node, launch_conf);

    // Set launch engine.
    XrtEngine engine = GraphEngine(node->sub_graph());
    launch_conf->set_engine([&]() -> std::string {
      switch (engine) {
        case XrtEngine::XLA: return "XLA";
        case XrtEngine::TENSORRT: return "TENSORRT";
        default: LOG(FATAL) << "Not supported engine " << engine; return "";
      }
    }());

    util::Set<std::string> input_mutability;
    bool is_model_update = false;
    // Build function, returns inputs mutabilities and is_model_update.
    buildFunction(node->sub_graph(), engine, &input_mutability, &is_model_update,
                  launch_conf->mutable_function());
    // Mark the launch op whether it is model update op or not.
    launch_conf->set_model_update(is_model_update);

    for (const auto &arg_proto : launch_conf->function().argument()) {
      std::string arg_value = arg_proto.value();
      const auto &it = fixedup_names_.find(arg_value);
      if (it != fixedup_names_.end()) { arg_value = it->second /* fixedup blob names */; }
      if (input_mutability.count(arg_proto.value()) > 0) {
        (*launch_conf->mutable_input_mutability())[arg_value] = true;
      }

      // Set input and output mapping from launch op to function.
      (*launch_conf->mutable_input_output_mapping())[arg_value] = arg_proto.value();
    }

    const auto& lbn2logical_blob_desc_map = builder_->job().helper().lbn2logical_blob_desc();
    for (const XrtNode *sub_node : node->sub_graph()->Nodes()) {
      for (const XrtEdge *edge : sub_node->in_edges()) {
        const Argument &argument = edge->argument();
        const std::string lbn = argument.name();
        auto iter = lbn2logical_blob_desc_map.find(lbn);
        CHECK(iter != lbn2logical_blob_desc_map.end());
        (*launch_conf->mutable_lbn2logical_blob_desc())[lbn] = iter->second;
      }
      for (const XrtEdge *edge : sub_node->out_edges()) {
        const Argument &argument = edge->argument();
        const std::string lbn = argument.name();
        auto iter = lbn2logical_blob_desc_map.find(lbn);
        CHECK(iter != lbn2logical_blob_desc_map.end());
        (*launch_conf->mutable_lbn2logical_blob_desc())[lbn] = iter->second;
      }
    }

    CHECK_GT(folded_nodes_[i].size(), 0);
    const ParallelConf &parallel_conf = builder_->ParallelConf4OpName(folded_nodes_[i][0]->name());
    // TODO(hjchen2) check parallel conf over all folded nodes

    builder_->AddOps(parallel_conf, {op_conf});
  }
}

void FoldSubgraphBuilder::FixupControlInOpNames() {
  CHECK_EQ(launch_nodes_.size(), folded_nodes_.size());
  // Map folded node names to cluster node
  util::Map<std::string, const XrtNode *> folded_op_names;
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    for (const XrtNode *node : folded_nodes_[i]) {
      folded_op_names.emplace(node->name(), launch_nodes_[i]);
    }
  }

  auto AddControlInOpName = [&](OperatorConf *conf, const std::string &op_name) -> void {
    std::string ctrl_in_op_name = op_name;
    const auto &it = folded_op_names.find(op_name);
    if (it != folded_op_names.end()) { ctrl_in_op_name = it->second->name(); }
    if (conf->name() != ctrl_in_op_name) {
      DoNoDuplicationAdd(conf->mutable_ctrl_in_op_name(), ctrl_in_op_name);
    }
  };

  for (const XrtNode *node : graph_.Nodes()) {
    auto *op_conf = builder_->MutableOpConf4OpName(node->name());
    if (node->sub_graph() == nullptr) {
      auto ctrl_in_op_names = op_conf->ctrl_in_op_name();
      op_conf->clear_ctrl_in_op_name();
      for (const auto &op_name : ctrl_in_op_names) { AddControlInOpName(op_conf, op_name); }
    } else {
      for (const XrtNode *sub_node : node->sub_graph()->Nodes()) {
        if (sub_node->IsArgumentNode()) { continue; }
        const auto &folded_op_conf = builder_->OpConf4OpName(sub_node->name());
        for (const auto &op_name : folded_op_conf.ctrl_in_op_name()) {
          AddControlInOpName(op_conf, op_name);
        }
      }
    }
  }
}

void FoldSubgraphBuilder::FixupInOutBlobNames() {
  for (const XrtNode *node : launch_nodes_) {
    std::string launch_op_name = node->name();
    // Fix input arguments consume key.
    util::Map<std::string, int> consume_argument_names;
    for (XrtEdge *edge : node->in_edges()) {
      if (edge->IsControlEdge()) { continue; }
      const Argument &arg = edge->argument();
      int index = consume_argument_names.size();
      auto it = consume_argument_names.find(arg.name());
      if (it == consume_argument_names.end()) {
        it = consume_argument_names.emplace(arg.name(), index).first;
      }
      index = it->second;

      ArgumentMetaData metadata;
      metadata.consume_key = absl::StrCat("in_", index);
      metadata.produce_key = arg.meta_data().produce_key;
      Argument fixed_arg(arg.name(), arg.shape(), arg.data_type(), metadata);
      edge->SetArgument(fixed_arg);
    }

    // Fix output blob names.
    util::Map<std::string, int> produce_argument_names;
    for (XrtEdge *edge : node->out_edges()) {
      if (edge->IsControlEdge()) { continue; }
      const Argument &arg = edge->argument();
      int index = produce_argument_names.size();
      auto it = produce_argument_names.find(arg.name());
      if (it == produce_argument_names.end()) {
        CHECK_EQ(fixedup_names_.count(arg.name()), 0);
        it = produce_argument_names.emplace(arg.name(), index).first;
      }
      index = it->second;

      std::string fixed_blob_name = absl::StrCat(launch_op_name, "/out_", index);
      fixedup_names_.emplace(arg.name(), fixed_blob_name);
      // Fix end input blob name
      const XrtNode *end = edge->end();
      if (end->type() != _XrtLaunchOpType) {
        auto *op_conf = builder_->MutableOpConf4OpName(end->name());
        const std::string &consume_key = arg.meta_data().consume_key;
        SetOpInputBlobName(op_conf, consume_key, arg.name(), fixed_blob_name);
      }
      ArgumentMetaData metadata;
      metadata.consume_key = arg.meta_data().consume_key;
      metadata.produce_key = absl::StrCat("out_", index);
      Argument fixed_arg(fixed_blob_name, arg.shape(), arg.data_type(), metadata);
      edge->SetArgument(fixed_arg);
    }
  }
}

void FoldSubgraphBuilder::FixupTimeShapes() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    CHECK_GT(folded_nodes_[i].size(), 0);
    const OpTimeShape &time_shape = builder_->TimeShape4OpName(folded_nodes_[i][0]->name());
    // TODO(hjchen2) check time shape for all folded nodes
    builder_->AddTimeShape4OpName(launch_nodes_[i]->name(), time_shape);
  }
}

void FoldSubgraphBuilder::FixupSbpSignatures() {
  for (const XrtNode *node : launch_nodes_) {
    SbpSignature sbp_conf;
    auto *sbp_parallel = sbp_conf.mutable_bn_in_op2sbp_parallel();
    for (const XrtEdge *edge : node->in_edges()) {
      CHECK(edge->HasAttr("sbp_policy"));
      const std::string &bn = edge->argument().meta_data().consume_key;
      (*sbp_parallel)[bn] = edge->Attr<std::vector<SbpParallel>>("sbp_policy")[1];
    }
    for (const XrtEdge *edge : node->out_edges()) {
      CHECK(edge->HasAttr("sbp_policy"));
      const std::string &bn = edge->argument().meta_data().produce_key;
      (*sbp_parallel)[bn] = edge->Attr<std::vector<SbpParallel>>("sbp_policy")[0];
    }
    // Append sbp signatures to helper
    builder_->AddSbpSignature4OpName(node->name(), sbp_conf);

    // Add function node sbp signatures.
    auto *op_conf = builder_->MutableOpConf4OpName(node->name());
    auto *launch_conf = op_conf->mutable_xrt_launch_conf();
    auto *sbp_signatures = launch_conf->mutable_sbp_signatures();
    for (const auto &node_conf : launch_conf->function().node()) {
      const std::string &node_name = node_conf.name();
      (*sbp_signatures)[node_name] = builder_->SbpSignature4OpName(node_name);
    }
  }
}

void FoldSubgraphBuilder::RemoveLaunchFoldedOps() {
  util::Set<std::string> removing_names;
  for (const XrtNode *node : launch_nodes_) {
    for (const XrtNode *sub_node : node->sub_graph()->Nodes()) {
      if (!sub_node->IsArgumentNode()) { removing_names.insert(sub_node->name()); }
    }
  }
  builder_->RemoveOpByName(removing_names);
}

// Rebuild job according to the nodes folded xrt graph. In order to rebuild
// the job, We will add several launch operators in the job, and remove the
// folded operators. In each launch operator, we wll reconstruct the subgraph
// and insert argument nodes if necessary.
class RebuildCompiledJobPass : public XrtPass {
 public:
  RebuildCompiledJobPass() = default;

  // params: vector of any which should contains:
  //   0 - job
  void Run(XrtGraph *graph, const XrtPassOptions &options,
           const std::vector<Any> &params) override {
    CHECK_GE(params.size(), 1) << "Job is required by `RebuildCompiledJobPass`.";
    auto *job = any_cast<Job *>(params[0]);

    CHECK(graph) << "Graph is required by `RebuildCompiledJobPass`.";
    FoldSubgraphBuilder(*graph, job).Build();
  }
};

REGISTER_XRT_PASS(RebuildCompiledJob, RebuildCompiledJobPass);

}  // namespace xrt
}  // namespace oneflow
