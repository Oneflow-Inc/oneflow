#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/rebuild_job.h"

namespace oneflow {

using XlaNode = mola::XlaNode;
using XlaEdge = mola::XlaEdge;
using Argument = mola::Argument;

namespace mola {
extern const std::string _XlaLaunchOpType;
extern const std::string _XlaArgumentOpType;
extern const std::string _XlaInArgumentPrefix;
extern const std::string _XlaOutArgumentPrefix;
}  // namespace mola

template <typename T>
using RepeatedPtrField = google::protobuf::RepeatedPtrField<T>;

template <typename T>
void DoNoDuplicationAdd(RepeatedPtrField<T> *repeat_field, const T &val) {
  if (std::find(repeat_field->begin(), repeat_field->end(), val) ==
      repeat_field->end()) {
    repeat_field->Add()->assign(val);
  }
}

XlaLaunchOpConf::Argument *MutableXlaArgumentProto(
                                XlaLaunchOpConf *launch_op_conf,
                                const std::string &op_name) {
  for (auto &proto : *(launch_op_conf->mutable_attr()->mutable_argument())) {
    if (proto.name() == op_name) {
      return &proto;
    }
  }
  return nullptr;
}

std::function<bool(const std::string &blob_name)> MakeHasBatchDimFunc(
        JobBuilder *builder) {
  auto has_batch_dim_fn = [builder](const std::string &blob_name) -> bool {
    const auto &batch_dim_lbis = builder->batch_dim_lbis();
    LogicalBlobId lbi = GenLogicalBlobId(blob_name);
    return batch_dim_lbis.count(lbi) > 0;
  };
  return has_batch_dim_fn;
}

std::function<SbpParallel(const std::string &blob_name)>
        MakeGetSbpSignatureFunc(JobBuilder *builder) {
  auto get_sbp_signature_fn = [builder](const std::string &blob_name) -> SbpParallel {
    LogicalBlobId lbi = GenLogicalBlobId(blob_name);
    const auto &sbp_signature =
        builder->OpSbpSignature(lbi.op_name()).bn_in_op2sbp_parallel();
    const auto &it = sbp_signature.find(lbi.blob_name());
    CHECK(it != sbp_signature.end());
    return it->second;
  };
  return get_sbp_signature_fn;
}

void FixSubgraphInArgumentsBlobNames(
        const mola::XlaGraph *sub_graph,
        const std::string &launch_op_name,
        XlaLaunchOpConf *launch_op_conf,
        const std::unordered_map<std::string, std::string> &fixed_blob_names) {
  for (const auto *node : sub_graph->Nodes()) {
    if (absl::StartsWith(node->op_name(), mola::_XlaInArgumentPrefix)) {
      auto *argument_conf = MutableXlaArgumentProto(launch_op_conf,
                                                    node->op_name());
      for (auto &blob_name : *(argument_conf->mutable_in())) {
        const auto &it = fixed_blob_names.find(blob_name);
        if (it != fixed_blob_names.end()) {
          blob_name = absl::StrCat(launch_op_name, "/", it->second);
        }
      }
    }
  }
}

void FixSubgraphOutArgumentsBlobNames(
        const mola::XlaGraph *sub_graph,
        XlaLaunchOpConf *launch_op_conf,
        const std::unordered_map<std::string, std::string> &fixed_blob_names) {
  for (const auto *node : sub_graph->Nodes()) {
    if (absl::StartsWith(node->op_name(), mola::_XlaOutArgumentPrefix)) {
      auto *argument_conf = MutableXlaArgumentProto(launch_op_conf,
                                                    node->op_name());
      for (auto &blob_name : *(argument_conf->mutable_out())) {
        const auto &it = fixed_blob_names.find(blob_name);
        if (it != fixed_blob_names.end()) {
          blob_name = it->second;
        }
      }
    }
  }
}

void FixControlInOpNames(const mola::XlaGraph &graph, JobBuilder *builder) {
  // Map folded node names to cluster node
  std::unordered_map<std::string, std::string> folded_op_names;

  for (const XlaNode *node : graph.Nodes()) {
    if (node->sub_graph() == nullptr) {
      continue;
    }
    std::string launch_op_name = node->op_name();
    for (const XlaNode *sub_node : node->sub_graph()->Nodes()) {
      folded_op_names.emplace(sub_node->op_name(), launch_op_name);
    }
  }

  auto AddControlInOpName = [&](OperatorConf *conf,
                                const std::string &op_name) -> void {
    std::string ctrl_in_op_name = op_name;
    const auto &it = folded_op_names.find(op_name);
    if (it != folded_op_names.end()) {
      ctrl_in_op_name = it->second;
    }
    DoNoDuplicationAdd(conf->mutable_ctrl_in_op_name(), ctrl_in_op_name);
  };

  for (const XlaNode *node : graph.Nodes()) {
    auto *op_conf = builder->MutableOpConf(node->op_name());
    if (node->sub_graph() == nullptr) {
      auto ctrl_in_op_names = op_conf->ctrl_in_op_name();
      op_conf->clear_ctrl_in_op_name();
      for (const auto &op_name : ctrl_in_op_names) {
        AddControlInOpName(op_conf, op_name);
      }
    } else {
      for (const XlaNode *sub_node : node->sub_graph()->Nodes()) {
        if (sub_node->op_type() == mola::_XlaArgumentOpType) {
          continue;
        }
        const auto &folded_op_conf = builder->OpConf(sub_node->op_name());
        for (const auto &op_name : folded_op_conf.ctrl_in_op_name()) {
          AddControlInOpName(op_conf, op_name);
        }
      }
    }
  }
}

void FixInOutBlobNames(const mola::XlaGraph &graph, JobBuilder *builder) {
  for (const XlaNode *node : graph.Nodes()) {
    if (node->sub_graph() == nullptr) {
      continue;
    }
    std::string launch_op_name = node->op_name();
    auto *launch_op_conf = builder->MutableOpConf(launch_op_name)
                                  ->mutable_xla_launch_conf();
    std::unordered_map<std::string, std::string> fixed_blob_names;
    // Fix output blob names
    for (int i = 0; i < launch_op_conf->out().size(); ++i) {
      std::string blob_name = launch_op_conf->out()[i];
      std::vector<std::string> name_split = absl::StrSplit(blob_name, "/");
      CHECK_EQ(name_split.size(), 2);
      std::string fixed_blob_name = absl::StrCat(name_split[1], "_", i);
      launch_op_conf->mutable_out()->Mutable(i)->assign(fixed_blob_name);
      fixed_blob_names.emplace(blob_name, fixed_blob_name);
    }
    // Infect changes to the next operators
    for (const XlaEdge *edge : node->out_edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }
      const XlaNode *end = edge->end();
      if (end->op_type() == mola::_XlaLaunchOpType) {
        auto *launch_op_conf = builder->MutableOpConf(end->op_name())
                                      ->mutable_xla_launch_conf();
        for (auto &blob_name : *launch_op_conf->mutable_in()) {
          const auto &it = fixed_blob_names.find(blob_name);
          if (it != fixed_blob_names.end()) {
            std::string fixed_blob_name = absl::StrCat(launch_op_name, "/",
                                                       it->second);
            blob_name = fixed_blob_name;
          }
        }
        // Fix input argument blob name for subgraph first
        FixSubgraphInArgumentsBlobNames(end->sub_graph(), launch_op_name,
                                        launch_op_conf, fixed_blob_names);
      } else {
        // TODO(hjchen2) Current implementation is ugly
        auto *op_conf = builder->MutableOpConf(end->op_name());
        for (const std::string &input : end->op()->input_bns()) {
          const LogicalBlobId &lbi = end->op()->BnInOp2Lbi(input);
          std::string blob_name = GenLogicalBlobName(lbi);
          const auto &it = fixed_blob_names.find(blob_name);
          if (it != fixed_blob_names.end()) {
            std::string fixed_blob_name = absl::StrCat(launch_op_name, "/",
                                                       it->second);
            // Fix input blob name for normal node
            auto *spec_conf = MutableMessageInPbMessage(
                                    op_conf, op_conf->op_type_case());
            SetBnValInOpTypeConf(spec_conf, input, blob_name, fixed_blob_name);
          }
        }
      }
    }
    // Subgraph output argument blob name
    FixSubgraphOutArgumentsBlobNames(node->sub_graph(), launch_op_conf,
                                     fixed_blob_names);
  }
}

void RemoveFoldedOperators(const mola::XlaGraph &graph,
                           JobBuilder *builder) {
  for (const XlaNode *node : graph.Nodes()) {
    if (node->sub_graph() != nullptr) {
      for (const XlaNode *sub_node : node->sub_graph()->Nodes()) {
        if (sub_node->op_type() != mola::_XlaArgumentOpType) {
          builder->RemoveOp(sub_node->op_name());
        }
      }
    }
  }
}

void FixSbpSignatures(const mola::XlaGraph &graph,
                      JobBuilder *builder) {
  auto get_sbp_signature_fn = MakeGetSbpSignatureFunc(builder);
  for (const XlaNode *node : graph.Nodes()) {
    if (node->sub_graph() == nullptr) {
      continue;
    }
    OperatorConf *op_conf = builder->MutableOpConf(node->op_name());
    std::shared_ptr<Operator> op = ConstructOp(*op_conf);
    std::unordered_map<std::string, std::string> input_blob_names;
    for (const std::string &bn : op->input_bns()) {
      std::string blob_name = GenLogicalBlobName(op->BnInOp2Lbi(bn));
      input_blob_names.emplace(blob_name, bn);
    }

    SbpSignature sbp_conf;
    auto *bn2sbp_parallel = sbp_conf.mutable_bn_in_op2sbp_parallel();
    const auto &attr_proto = op_conf->xla_launch_conf().attr();
    for (const auto &argument : attr_proto.argument()) {
      // Input argument
      if (absl::StartsWith(argument.name(), mola::_XlaInArgumentPrefix)) {
        CHECK_EQ(argument.in().size(), argument.out().size());
        for (int i = 0; i < argument.in().size(); ++i) {
          std::string bn = input_blob_names.at(argument.in()[i]);
          (*bn2sbp_parallel)[bn] = get_sbp_signature_fn(argument.out()[i]);
        }
      }
      // Output argument
      if (absl::StartsWith(argument.name(), mola::_XlaOutArgumentPrefix)) {
        CHECK_EQ(argument.in().size(), argument.out().size());
        for (int i = 0; i < argument.in().size(); ++i) {
          std::string bn = argument.out()[i];
          (*bn2sbp_parallel)[bn] = get_sbp_signature_fn(argument.in()[i]);
        }
      }
    }
    // Add the Sbp Signature to the job
    builder->AddOpSbpSignature(node->op_name(), sbp_conf);
  }
}

void buildXlaLaunchAttribute(const mola::XlaGraph *graph,
                             JobBuilder *builder,
                             XlaLaunchOpConf::Attribute *launch_attr) {
  auto has_batch_dim_fn = MakeHasBatchDimFunc(builder);
  for (const XlaNode *node : graph->Nodes()) {
    if (node->op_type() != mola::_XlaArgumentOpType) {
      *(launch_attr->add_node()) = node->op()->op_conf();      
    } else {
      XlaLaunchOpConf::Argument argument_proto;
      argument_proto.set_name(node->op_name());
      // Usually one argument node has either inputs or outputs
      CHECK(node->in_edges().size() == 0 || node->out_edges().size() == 0);
      // Build inputs or outputs for the argument nodes
      for (const XlaEdge *edge : node->out_edges()) {
        const Argument &argument = edge->argument();
        DoNoDuplicationAdd(argument_proto.mutable_in(),
                           argument.blob_name());
        DoNoDuplicationAdd(argument_proto.mutable_out(),
                           argument.blob_name());
      }
      for (const XlaEdge *edge : node->in_edges()) {
        const Argument &argument = edge->argument();
        DoNoDuplicationAdd(argument_proto.mutable_in(),
                           argument.blob_name());
        DoNoDuplicationAdd(argument_proto.mutable_out(),
                           argument.blob_name());
      }
      // Restore the blob names that have batch dimension, so that it's no need
      // to infer `HasBatchDim` for `XlaLaunch` operators. In practice it's hard
      // to infer `HasBatchDim` since the operators to be infered have been
      // folded. Normally we have to infer `HasBatchDim` before `SbpSignature`
      // and `BlobDesc`, and `HasBatchDim` will reply on the front operators
      // `BlobDesc`. Therefor we probably could not infer `HasBatchDim` for the
      // folded operators because their inputs `BlobDesc` were not infered if
      // the front operators have been folded as well
      for (const std::string &blob_name : argument_proto.out()) {
        if (has_batch_dim_fn(blob_name)) {
          DoNoDuplicationAdd(launch_attr->mutable_batch_dim_blob(),
                             blob_name);
        }
      }
      *(launch_attr->add_argument()) = argument_proto;
    }
  }
}

void RebuildXlaCompiledJob(const mola::XlaGraph &graph, Job *job) {
  JobBuilder builder(job);

  for (const XlaNode *node : graph.Nodes()) {
    if (node->op_type() != mola::_XlaLaunchOpType) {
      continue;
    }
    // Add xla launch operator
    OperatorConf op_conf; 
    op_conf.set_name(node->op_name());

    XlaLaunchOpConf *launch_op_conf = op_conf.mutable_xla_launch_conf();
    for (const XlaEdge *edge : node->in_edges()) {
      if (!edge->IsControlEdge()) {
        const Argument &arg = edge->argument();
        DoNoDuplicationAdd(launch_op_conf->mutable_in(), arg.blob_name());
      }
    }
    for (const XlaEdge *edge : node->out_edges()) {
      if (!edge->IsControlEdge()) {
        const Argument &arg = edge->argument();
        DoNoDuplicationAdd(launch_op_conf->mutable_out(), arg.blob_name());
      }
    }

    XlaLaunchOpConf::Attribute *launch_attr = launch_op_conf->mutable_attr();
    buildXlaLaunchAttribute(node->sub_graph(), &builder, launch_attr);
   
    // TODO(hjchen2) Assign parallel conf 
    ParallelConf parallel_conf;
    parallel_conf.set_policy(kDataParallel);
    parallel_conf.mutable_device_name()->Add()->assign("0:gpu:0");
    builder.AddOps(parallel_conf, {op_conf});
  }

  // Fix ctrl_in_op_name
  FixControlInOpNames(graph, &builder);
  // Fix blob names
  FixInOutBlobNames(graph, &builder);
  // Fix Sbp signatures
  FixSbpSignatures(graph, &builder);
  // Remove folded operator conf from the job
  RemoveFoldedOperators(graph, &builder);
}

}  // namespace oneflow
