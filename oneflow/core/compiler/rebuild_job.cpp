#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/rebuild_job.h"

namespace oneflow {

using XlaNode = mola::XlaNode;
using XlaEdge = mola::XlaEdge;
using Argument = mola::Argument;
using XlaGraph = mola::XlaGraph;

namespace mola {
extern const std::string _XlaLaunchOpType;
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

class FoldSubgraphBuilder {
 public:
  FoldSubgraphBuilder(const XlaGraph &graph, Job *job);

  virtual ~FoldSubgraphBuilder() {}

  void Build() { BuildImpl(); }
  
 private:
  void BuildImpl() {
    CHECK(builder_);
    // Rebuilding folded job should takes next steps in order
    // 1. Add XlaLaunch operator in the job
    BuildXlaLaunchOps();
    // 2. Replace control_in_op_name by XlaLaunch name if the operator
    //    has been folded by a XlaLaunch operator
    FixupControlInOpNames();
    // 3. Fixup input blob names for all operators if it's input has
    //    been folded, and fixup it's outputs
    FixupInOutBlobNames();
    // 4. Add time shape for the XlaLaunch operators
    FixupTimeShapes();
    // 5. Add sbp parallel strategy for the XlaLaunch operators
    FixupSbpSignatures();
    // 6. Finally remove the folded operators
    RemoveLaunchFoldedOps();
  }

  void buildXlaLaunchAttribute(const XlaGraph *sub_graph,
                               XlaLaunchOpConf::Attribute *launch_attr);

  void FixSubgraphInArgumentsBlobNames(
      const XlaGraph *sub_graph,
      const std::string &launch_op_name,
      XlaLaunchOpConf *launch_conf,
      const std::unordered_map<std::string, std::string> &fixed_blob_names);

  void FixSubgraphOutArgumentsBlobNames(
    const XlaGraph *sub_graph,
    const std::string &launch_op_name,
    XlaLaunchOpConf *launch_conf,
    const std::unordered_map<std::string, std::string> &fixed_blob_names);

  bool HasBatchDim(const std::string &blob_name) {
    const auto &batch_dim_lbis = builder_->batch_dim_lbis();
    LogicalBlobId lbi = BlobId(blob_name);
    return batch_dim_lbis.count(lbi) > 0;
  }
  
  XlaLaunchOpConf::Argument *MutableArgumentConf(
      XlaLaunchOpConf *launch_conf, const std::string &argument_name) {
    XlaLaunchOpConf::Argument *argument_conf = nullptr;
    XlaLaunchOpConf::Attribute *attr = launch_conf->mutable_attr();
    for (auto &argument_proto : *(attr->mutable_argument())) {
      if (argument_proto.name() == argument_name) {
        argument_conf = &argument_proto;
        break;
      }
    }
    return argument_conf;
  }

  SbpParallel GetSbpSignature(const std::string &blob_name) {
    LogicalBlobId lbi = BlobId(blob_name);
    const auto &sbp_signatures =
          builder_->GetSbpSignature(lbi.op_name()).bn_in_op2sbp_parallel();
    const auto &it = sbp_signatures.find(lbi.blob_name());
    CHECK(it != sbp_signatures.end());
    return it->second;
  }

  void BuildXlaLaunchOps();

  void FixupControlInOpNames();

  void FixupInOutBlobNames();

  void FixupTimeShapes();

  void FixupSbpSignatures();

  void RemoveLaunchFoldedOps();

 private:
  const XlaGraph &graph_;
  std::shared_ptr<JobBuilder> builder_;
  // Launch nodes
  std::vector<const XlaNode *> launch_nodes_;
  // Folded nodes except for argument nodes for each launch nodes
  std::vector<std::vector<const XlaNode *>> folded_nodes_;
};

FoldSubgraphBuilder::FoldSubgraphBuilder(const XlaGraph &graph, Job *job)
    : graph_(graph) {
  for (const XlaNode *node : graph_.Nodes()) {
    if (node->op_type() == mola::_XlaLaunchOpType) {
      launch_nodes_.push_back(node);
    }
  }

  folded_nodes_.resize(launch_nodes_.size());
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    XlaGraph *sub_graph = launch_nodes_[i]->sub_graph();
    CHECK_NOTNULL(sub_graph);
    for (const XlaNode *sub_node : sub_graph->Nodes()) {
      if (!sub_node->IsArgumentNode()) {
        folded_nodes_[i].push_back(sub_node);
      }
    }
  }
  builder_ = std::make_shared<JobBuilder>(job);
}

void FoldSubgraphBuilder::FixSubgraphInArgumentsBlobNames(
    const XlaGraph *sub_graph,
    const std::string &launch_op_name,
    XlaLaunchOpConf *launch_conf,
    const std::unordered_map<std::string, std::string> &fixed_blob_names) {
  for (const auto *node : sub_graph->Nodes()) {
    if (node->IsInArgumentNode()) {
      auto *argument_conf = MutableArgumentConf(launch_conf, node->op_name());
      const auto &it = fixed_blob_names.find(argument_conf->in());
      if (it != fixed_blob_names.end()) {
        argument_conf->set_in(absl::StrCat(launch_op_name, "/", it->second));
      }
    }
  }
}

void FoldSubgraphBuilder::FixSubgraphOutArgumentsBlobNames(
    const XlaGraph *sub_graph,
    const std::string &launch_op_name,
    XlaLaunchOpConf *launch_conf,
    const std::unordered_map<std::string, std::string> &fixed_blob_names) {
  for (const auto *node : sub_graph->Nodes()) {
    if (node->IsOutArgumentNode()) {
      auto *argument_conf = MutableArgumentConf(launch_conf, node->op_name());
      const auto &it = fixed_blob_names.find(argument_conf->out());
      if (it != fixed_blob_names.end()) {
        // argument_conf->set_out(absl::StrCat(launch_op_name, "/", it->second));
        argument_conf->set_out(it->second);
      }
    }
  }
}

void FoldSubgraphBuilder::buildXlaLaunchAttribute(
    const XlaGraph *sub_graph, XlaLaunchOpConf::Attribute *launch_attr) {
  for (const XlaNode *node : sub_graph->Nodes()) {
    if (!node->IsArgumentNode()) {
      *(launch_attr->add_node()) = node->op()->op_conf();      
    } else {
      auto *argument_proto = launch_attr->add_argument();
      argument_proto->set_name(node->op_name());
      // Usually one argument node has either inputs or outputs
      CHECK(node->in_edges().size() == 0 || node->out_edges().size() == 0);
      // Build inputs or outputs for the argument nodes
      for (const XlaEdge *edge : node->out_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_in(argument.blob_name());
        argument_proto->set_out(argument.blob_name());
      }
      for (const XlaEdge *edge : node->in_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_in(argument.blob_name());
        argument_proto->set_out(argument.blob_name());
      }

      // Restore the blob names that have batch dimension, so that it's no need
      // to infer `HasBatchDim` for `XlaLaunch` operators. In practice it's hard
      // to infer `HasBatchDim` since the operators to be infered have been
      // folded. Normally we have to infer `HasBatchDim` before `SbpSignature`
      // and `BlobDesc`, and `HasBatchDim` replies on the front operators
      // `BlobDesc`. Therefor we probably could not infer `HasBatchDim` for the
      // folded operators because their inputs `BlobDesc` were not infered since
      // the front operators have been folded as well
      if (HasBatchDim(argument_proto->out())) {
        DoNoDuplicationAdd(launch_attr->mutable_batch_dim_blob(),
                           argument_proto->out());
      }
    }
  }
}

void AddInBlobNames(const std::list<XlaEdge *> &in_edges,
                    XlaLaunchOpConf *launch_conf) {
  for (const XlaEdge *edge : in_edges) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      DoNoDuplicationAdd(launch_conf->mutable_in(), arg.blob_name());
    }
  }
}

void AddOutBlobNames(const std::list<XlaEdge *> &out_edges,
                    XlaLaunchOpConf *launch_conf) {
  for (const XlaEdge *edge : out_edges) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      DoNoDuplicationAdd(launch_conf->mutable_out(), arg.blob_name());
    }
  }
}

void FoldSubgraphBuilder::BuildXlaLaunchOps() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    const XlaNode *node = launch_nodes_[i];
    // Add xla launch operator
    OperatorConf op_conf; 
    op_conf.set_name(node->op_name());

    XlaLaunchOpConf *launch_conf = op_conf.mutable_xla_launch_conf();
    AddInBlobNames(node->in_edges(), launch_conf);
    AddOutBlobNames(node->out_edges(), launch_conf);

    buildXlaLaunchAttribute(node->sub_graph(), launch_conf->mutable_attr());

    CHECK_GT(folded_nodes_[i].size(), 0);
    ParallelConf parallel_conf = builder_->GetParallelConf(
        folded_nodes_[i][0]->op_name());
    // TODO(hjchen2) check parallel conf over all folded nodes

    builder_->AddOps(parallel_conf, {op_conf});
  }
}


void FoldSubgraphBuilder::FixupControlInOpNames() {
  CHECK_EQ(launch_nodes_.size(), folded_nodes_.size());
  // Map folded node names to cluster node
  std::unordered_map<std::string, const XlaNode *> folded_op_names;
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    for (const XlaNode *node : folded_nodes_[i]) {
      folded_op_names.emplace(node->op_name(), launch_nodes_[i]);
    }
  }

  auto AddControlInOpName = [&](OperatorConf *conf,
                                const std::string &op_name) -> void {
    std::string ctrl_in_op_name = op_name;
    const auto &it = folded_op_names.find(op_name);
    if (it != folded_op_names.end()) {
      ctrl_in_op_name = it->second->op_name();
    }
    DoNoDuplicationAdd(conf->mutable_ctrl_in_op_name(), ctrl_in_op_name);
  };

  for (const XlaNode *node : graph_.Nodes()) {
    auto *op_conf = builder_->MutableOpConf(node->op_name());
    if (node->sub_graph() == nullptr) {
      auto ctrl_in_op_names = op_conf->ctrl_in_op_name();
      op_conf->clear_ctrl_in_op_name();
      for (const auto &op_name : ctrl_in_op_names) {
        AddControlInOpName(op_conf, op_name);
      }
    } else {
      for (const XlaNode *sub_node : node->sub_graph()->Nodes()) {
        if (sub_node->IsArgumentNode()) {
          continue;
        }
        const auto &folded_op_conf = builder_->GetOpConf(sub_node->op_name());
        for (const auto &op_name : folded_op_conf.ctrl_in_op_name()) {
          AddControlInOpName(op_conf, op_name);
        }
      }
    }
  }
}

void FoldSubgraphBuilder::FixupInOutBlobNames() {
  for (const XlaNode *node : launch_nodes_) {
    std::string launch_op_name = node->op_name();
    auto *launch_conf = builder_->MutableOpConf(launch_op_name)
                                ->mutable_xla_launch_conf();
    std::unordered_map<std::string, std::string> fixed_blob_names;
    // Fix output blob names
    for (int i = 0; i < launch_conf->out().size(); ++i) {
      std::string blob_name = launch_conf->out()[i];
      std::string fixed_blob_name = absl::StrCat("out_", i);
      fixed_blob_names.emplace(blob_name, fixed_blob_name);

      launch_conf->mutable_out()->Mutable(i)->assign(fixed_blob_name);
      // Append to `batch_dim_lbis`
      if (HasBatchDim(blob_name)) {
        LogicalBlobId lbi;
        lbi.set_op_name(launch_op_name);
        lbi.set_blob_name(fixed_blob_name);
        builder_->AddBatchDimLbi(lbi);
      }
    }

    // Infect changes to the next operators only once
    std::unordered_set<const XlaNode *> changed_nodes;

    for (const XlaEdge *edge : node->out_edges()) {
      const XlaNode *end = edge->end();
      if (edge->IsControlEdge() || !changed_nodes.insert(end).second) {
        continue;
      }
      if (end->op_type() == mola::_XlaLaunchOpType) {
        auto *launch_conf = builder_->MutableOpConf(end->op_name())
                                    ->mutable_xla_launch_conf();
        for (auto &blob_name : *launch_conf->mutable_in()) {
          const auto &it = fixed_blob_names.find(blob_name);
          if (it != fixed_blob_names.end()) {
            std::string fixed_blob_name = absl::StrCat(launch_op_name, "/",
                                                       it->second);
            blob_name = fixed_blob_name;
          }
        }
        // Fix input argument blob name for subgraph first
        FixSubgraphInArgumentsBlobNames(end->sub_graph(), launch_op_name,
                                        launch_conf, fixed_blob_names);
      } else {
        // TODO(hjchen2) Current implementation is ugly
        auto *op_conf = builder_->MutableOpConf(end->op_name());
        for (const std::string &input : end->op()->input_bns()) {
          const LogicalBlobId &lbi = end->op()->BnInOp2Lbi(input);
          std::string blob_name = BlobName(lbi);
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
    // Fix subgraph output argument blob name
    FixSubgraphOutArgumentsBlobNames(node->sub_graph(), launch_op_name,
                                     launch_conf, fixed_blob_names);
  }
}

void FoldSubgraphBuilder::FixupTimeShapes() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    CHECK_GT(folded_nodes_[i].size(), 0);
    OpTimeShape time_shape = builder_->GetTimeShape(
        folded_nodes_[i][0]->op_name());
    // TODO(hjchen2) check time shape for all folded nodes

    builder_->AddTimeShape(launch_nodes_[i]->op_name(), time_shape);
  }
}

void FoldSubgraphBuilder::FixupSbpSignatures() {
  for (const XlaNode *node : launch_nodes_) {
    OperatorConf *op_conf = builder_->MutableOpConf(node->op_name());
    std::shared_ptr<Operator> op = ConstructOp(*op_conf);
    std::unordered_map<std::string, std::string> blob_names;
    for (const std::string &bn : op->input_bns()) {
      std::string blob_name = BlobName(op->BnInOp2Lbi(bn));
      blob_names.emplace(blob_name, bn);
    }

    SbpSignature sbp_conf;
    auto *sbp_signatures = sbp_conf.mutable_bn_in_op2sbp_parallel();
    auto *attr_proto = op_conf->mutable_xla_launch_conf()->mutable_attr();
    for (const auto &argument : attr_proto->argument()) {
      // Input argument
      if (absl::StartsWith(argument.name(), mola::_XlaInArgumentPrefix)) {
        std::string bn = blob_names.at(argument.in());
        (*sbp_signatures)[bn] = GetSbpSignature(argument.out());
      }
      // Output argument
      if (absl::StartsWith(argument.name(), mola::_XlaOutArgumentPrefix)) {
        std::string bn = argument.out();
        (*sbp_signatures)[bn] = GetSbpSignature(argument.in());
      }
    }
    // Append sbp signatures to xla launch operator
    *(attr_proto->mutable_sbp_signature()) = *sbp_signatures;
    // Append sbp signatures to helper
    builder_->AddSbpSignature(node->op_name(), sbp_conf);
  }
}

void FoldSubgraphBuilder::RemoveLaunchFoldedOps() {
  for (const XlaNode *node : launch_nodes_) {
    for (const XlaNode *sub_node : node->sub_graph()->Nodes()) {
      if (!sub_node->IsArgumentNode()) {
        builder_->RemoveOp(sub_node->op_name());
      }
    }
  }
}

void RebuildXlaCompiledJob(const XlaGraph &graph, Job *job) {
  FoldSubgraphBuilder builder(graph, job);
  builder.Build();
}

}  // namespace oneflow
