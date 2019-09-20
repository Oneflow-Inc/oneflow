#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_argument.h"
#include "oneflow/xla/rebuild_job.h"

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

int GetRepeatedIndex(const std::string &input) {
  std::vector<std::string> splits = absl::StrSplit(input, "_");
  CHECK_GT(splits.size(), 0);
  int index = 0;
  absl::SimpleAtoi(splits.back(), &index);
  return index;
};

void SetOpInputBlobName(OperatorConf *op_conf, const std::string &input,
                        const std::string &blob_name,
                        const std::string &fixed_blob_name) {
  auto *spec_conf = MutableMessageInPbMessage(
                          op_conf, op_conf->op_type_case());
  switch (op_conf->op_type_case()) {
    case OperatorConf::kPrintConf: {
      int index = GetRepeatedIndex(input);
      *(op_conf->mutable_print_conf()->mutable_in(index)->mutable_lbn())
          = fixed_blob_name;
      break;
    }
    default:
      ReplaceStrValInPbFdOrPbRpf(spec_conf, input, blob_name, fixed_blob_name);
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
    // Rebuilding folded job should takes the below steps in order
    // 1.Add XlaLaunch operator in the job
    BuildXlaLaunchOps();
    // 2.Replace control_in_op_name by XlaLaunch name if the operator
    //   has been folded by a XlaLaunch operator
    FixupControlInOpNames();
    // 3.Add time shape for the XlaLaunch operators
    FixupTimeShapes();
    // 4.Add sbp parallel strategy for the XlaLaunch operators
    FixupSbpSignatures();
    // 5.Fixup input blob names for all operators if it's input has
    //   been folded, and fixup it's outputs
    FixupInOutBlobNames();
    // 6.Finally remove the folded operators
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

  void BuildXlaLaunchOps();

  void FixupControlInOpNames();

  void FixupTimeShapes();

  void FixupSbpSignatures();

  void FixupInOutBlobNames();

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
  auto *resource_scope = launch_attr->mutable_resource_scope();
  for (const XlaNode *node : sub_graph->Nodes()) {
    if (!node->IsArgumentNode()) {
      *(launch_attr->add_node()) = node->op()->op_conf();
    } else {
      auto *argument_proto = launch_attr->add_argument();
      argument_proto->set_name(node->op_name());
      DeviceType device_type = mola::BackendToDeviceType(node->backend());
      argument_proto->set_device_type(device_type);
      // Usually one argument node has either inputs or outputs
      CHECK(node->in_edges().size() == 0 || node->out_edges().size() == 0);
      bool is_mutable = false;
      // Build inputs or outputs for the argument nodes
      for (const XlaEdge *edge : node->out_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_in(argument.blob_name());
        argument_proto->set_out(argument.blob_name());
        is_mutable |= IsMutableArgument(edge->end(), argument);
      }
      for (const XlaEdge *edge : node->in_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_in(argument.blob_name());
        argument_proto->set_out(argument.blob_name());
      }
      if (is_mutable) {
        argument_proto->set_is_mutable(true);
      }

      // Restore the batch axis that have batch dimension, so that it's no need
      // to infer `HasBatchAxis` for `XlaLaunch` operators. In practice it's hard
      // to infer `HasBatchAxis` since the operators to be infered have been
      // folded. Normally we have to infer `HasBatchAxis` before `SbpSignature`
      // and `BlobDesc`, and `HasBatchAxis` replies on the front operators
      // `BlobDesc`. Therefor we probably could not infer `HasBatchAxis` for the
      // folded operators because their inputs `BlobDesc` were not infered since
      // the front operators have been folded as well
      const std::string &argument_out = argument_proto->out();
      if (builder_->HasBatchAxis(argument_out)) {
        auto *batch_axis = resource_scope->mutable_batch_axis();
        (*batch_axis)[argument_out] = builder_->GetBatchAxis(argument_out);
      }
    }

    // Restore output shapes
    auto &shapes = *(resource_scope->mutable_shapes());
    for (const XlaEdge *edge : node->out_edges()) {
      const Argument &argument = edge->argument();
      const std::string blob_name = argument.blob_name();
      *(shapes[blob_name].mutable_shape()) = argument.shape_proto();
      shapes[blob_name].set_data_type(argument.data_type());
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
    DeviceType device_type = mola::BackendToDeviceType(node->backend());
    op_conf.set_device_type(device_type);

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
    if (conf->name() != ctrl_in_op_name) {
      DoNoDuplicationAdd(conf->mutable_ctrl_in_op_name(), ctrl_in_op_name);
    }
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
      // Append to `batch_axis`
      if (builder_->HasBatchAxis(blob_name)) {
        fixed_blob_name = absl::StrCat(launch_op_name, "/", fixed_blob_name);
        builder_->AddBatchAxis(fixed_blob_name,
                               builder_->GetBatchAxis(blob_name));
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
            SetOpInputBlobName(op_conf, input, blob_name, fixed_blob_name);
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
    std::shared_ptr<Operator> op = ConstructOp(*op_conf, &GlobalJobDesc());
    std::unordered_map<std::string, std::string> blob_names;
    for (const std::string &bn : op->input_bns()) {
      std::string blob_name = BlobName(op->BnInOp2Lbi(bn));
      blob_names.emplace(blob_name, bn);
    }
    for (const std::string &bn : op->output_bns()) {
      std::string blob_name = op->BnInOp2Lbi(bn).blob_name();
      blob_names.emplace(blob_name, bn);
    }

    SbpSignature sbp_conf;
    auto *sbp_signatures = sbp_conf.mutable_bn_in_op2sbp_parallel();
    auto *attr_proto = op_conf->mutable_xla_launch_conf()->mutable_attr();
    for (const XlaEdge *edge : node->in_edges()) {
      std::string bn = blob_names.at(edge->argument().blob_name());
      (*sbp_signatures)[bn] = edge->sbp_policy(1);
    }
    for (const XlaEdge *edge : node->out_edges()) {
      std::string bn = blob_names.at(edge->argument().blob_name());
      (*sbp_signatures)[bn] = edge->sbp_policy(0);
    }
    auto *resource_scope = attr_proto->mutable_resource_scope();
    // Append sbp signatures to xla launch operator
    *(resource_scope->mutable_sbp_signatures()) = *sbp_signatures;
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
  FoldSubgraphBuilder(graph, job).Build();
}

}  // namespace oneflow
