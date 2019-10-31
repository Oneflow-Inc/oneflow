#include <string>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

extern const std::string _XrtLaunchOpType;
extern const std::string _XrtInArgumentPrefix;
extern const std::string _XrtOutArgumentPrefix;

static const std::string _ReduceSplitType = "ReduceSplit";

template <typename T>
void DoNoDuplicationAdd(util::PbVector<T> *repeat_field, const T &val) {
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
  auto *spec_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
  switch (op_conf->op_type_case()) {
    case OperatorConf::kPrintConf: {
      int index = GetRepeatedIndex(input);
      *(op_conf->mutable_print_conf()->mutable_in(index)->mutable_lbn()) =
          fixed_blob_name;
      break;
    }
    default:
      ReplaceStrValInPbFdOrPbRpf(spec_conf, input, blob_name, fixed_blob_name);
  }
}

class FoldSubgraphBuilder {
 public:
  FoldSubgraphBuilder(const XrtGraph &graph, Job *job);

  virtual ~FoldSubgraphBuilder() {}

  void Build() {
    CHECK(builder_) << "Builder has not been initialized.";
    // Rebuilding folded job should takes the below steps in order
    InferIsAfterAllReduce();
    // 1.Add XrtLaunch operator to the job
    BuildXrtLaunchOps();
    // 2.Replace control_in_op_name by XrtLaunch name if the operator
    //   has been folded by a XrtLaunch operator
    FixupControlInOpNames();
    // 3.Add time shape for XrtLaunch operators
    FixupTimeShapes();
    // 4.Add sbp parallel strategy for XrtLaunch operators
    FixupSbpSignatures();
    // 5.Fixup input blob names for all operators if it's input has
    //   been folded, and fixup it's outputs
    FixupInOutBlobNames();
    // 6.Finally remove the folded operators
    RemoveLaunchFoldedOps();
  }

 private:
  void InferIsAfterAllReduce();

  void buildXrtLaunchAttribute(const XrtGraph *sub_graph,
                               XrtLaunchOpConf::Attribute *launch_attr);

  XrtLaunchOpConf::Argument *MutableArgumentConf(
      XrtLaunchOpConf *launch_conf, const std::string &argument_name) {
    XrtLaunchOpConf::Argument *argument_conf = nullptr;
    XrtLaunchOpConf::Attribute *attr = launch_conf->mutable_attr();
    for (auto &argument_proto : *(attr->mutable_argument())) {
      if (argument_proto.name() == argument_name) {
        argument_conf = &argument_proto;
        break;
      }
    }
    return argument_conf;
  }

  void FixSubgraphInArgumentsBlobNames(
      const XrtGraph *sub_graph, const std::string &launch_op_name,
      XrtLaunchOpConf *launch_conf,
      const util::Map<std::string, std::string> &fixed_blob_names);

  void FixSubgraphOutArgumentsBlobNames(
      const XrtGraph *sub_graph, const std::string &launch_op_name,
      XrtLaunchOpConf *launch_conf,
      const util::Map<std::string, std::string> &fixed_blob_names);

  void BuildXrtLaunchOps();

  void FixupControlInOpNames();

  void FixupTimeShapes();

  void FixupSbpSignatures();

  void FixupInOutBlobNames();

  void RemoveLaunchFoldedOps();

  bool IsAfterAllReduce(const XrtNode *node);

 private:
  const XrtGraph &graph_;
  std::shared_ptr<JobBuilder> builder_;
  // Launch nodes
  std::vector<const XrtNode *> launch_nodes_;
  // Folded nodes except for argument nodes for each launch nodes
  std::vector<std::vector<const XrtNode *>> folded_nodes_;

  std::unordered_set<const XrtNode *> after_allreduce_nodes_;
};

FoldSubgraphBuilder::FoldSubgraphBuilder(const XrtGraph &graph, Job *job)
    : graph_(graph) {
  for (const XrtNode *node : graph_.Nodes()) {
    if (node->type() == _XrtLaunchOpType) {
      launch_nodes_.push_back(node);
    }
  }

  folded_nodes_.resize(launch_nodes_.size());
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    XrtGraph *sub_graph = launch_nodes_[i]->sub_graph();
    CHECK_NOTNULL(sub_graph);
    for (const XrtNode *sub_node : sub_graph->Nodes()) {
      if (!sub_node->IsArgumentNode()) {
        folded_nodes_[i].push_back(sub_node);
      }
    }
  }
  builder_ = std::make_shared<JobBuilder>(job);
}

void FoldSubgraphBuilder::FixSubgraphInArgumentsBlobNames(
    const XrtGraph *sub_graph, const std::string &launch_op_name,
    XrtLaunchOpConf *launch_conf,
    const util::Map<std::string, std::string> &fixed_blob_names) {
  for (const auto *node : sub_graph->Nodes()) {
    if (node->IsInArgumentNode()) {
      auto *argument_conf = MutableArgumentConf(launch_conf, node->name());
      const auto &it = fixed_blob_names.find(argument_conf->in());
      if (it != fixed_blob_names.end()) {
        argument_conf->set_in(absl::StrCat(launch_op_name, "/", it->second));
      }
    }
  }
}

void FoldSubgraphBuilder::FixSubgraphOutArgumentsBlobNames(
    const XrtGraph *sub_graph, const std::string &launch_op_name,
    XrtLaunchOpConf *launch_conf,
    const util::Map<std::string, std::string> &fixed_blob_names) {
  for (const auto *node : sub_graph->Nodes()) {
    if (node->IsOutArgumentNode()) {
      auto *argument_conf = MutableArgumentConf(launch_conf, node->name());
      const auto &it = fixed_blob_names.find(argument_conf->out());
      if (it != fixed_blob_names.end()) {
        // argument_conf->set_out(absl::StrCat(launch_op_name, "/",
        // it->second));
        argument_conf->set_out(it->second);
      }
    }
  }
}

void FoldSubgraphBuilder::buildXrtLaunchAttribute(
    const XrtGraph *sub_graph, XrtLaunchOpConf::Attribute *launch_attr) {
  for (const XrtNode *node : sub_graph->Nodes()) {
    if (!node->IsArgumentNode()) {
      *(launch_attr->add_node()) =
          *reinterpret_cast<const OperatorConf *>(&node->param());
    } else {
      auto *argument_proto = launch_attr->add_argument();
      argument_proto->set_name(node->name());
      DeviceType device_type = XrtDeviceToDeviceType(node->device());
      argument_proto->set_device_type(device_type);
      // Usually one argument node has either inputs or outputs
      CHECK(node->in_edges().size() == 0 || node->out_edges().size() == 0);
      bool is_mutable = false;
      // Build inputs or outputs for the argument nodes
      for (const XrtEdge *edge : node->out_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_in(argument.name());
        argument_proto->set_out(argument.name());
        // is_mutable |= IsMutableArgument(edge->end(), argument);
      }
      for (const XrtEdge *edge : node->in_edges()) {
        const Argument &argument = edge->argument();
        argument_proto->set_in(argument.name());
        argument_proto->set_out(argument.name());
      }
      if (is_mutable) {
        (*launch_attr->mutable_mutability())[node->name()] = true;
        // argument_proto->set_is_mutable(true);
      }

      // Store the batch axis that have batch dimension, so that it's no need to
      // infer `HasBatchAxis4Lbn` for `XrtLaunch` operators. In practice it's
      // hard to infer `HasBatchAxis4Lbn` since the operators to be infered have
      // been folded. Normally we have to infer `HasBatchAxis4Lbn` before
      // `SbpSignature` and `BlobDesc`, and `HasBatchAxis4Lbn` replies on the
      // front operators `BlobDesc`. Therefor we probably could not infer
      // `HasBatchAxis4Lbn` for the folded operators because their inputs
      // `BlobDesc` were not infered since the front operators have been folded
      // as well
      const std::string &argument_out = argument_proto->out();
      if (builder_->HasBatchAxis4Lbn(argument_out)) {
        auto *batch_axis = launch_attr->mutable_batch_axis();
        (*batch_axis)[argument_out] = builder_->BatchAxis4Lbn(argument_out);
      }
    }
  }
}

void AddInBlobNames(const util::List<XrtEdge *> &in_edges,
                    XrtLaunchOpConf *launch_conf) {
  for (const XrtEdge *edge : in_edges) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      DoNoDuplicationAdd(launch_conf->mutable_in(), arg.name());
    }
  }
}

void AddOutBlobNames(const util::List<XrtEdge *> &out_edges,
                     XrtLaunchOpConf *launch_conf) {
  for (const XrtEdge *edge : out_edges) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      DoNoDuplicationAdd(launch_conf->mutable_out(), arg.name());
    }
  }
}

void FoldSubgraphBuilder::BuildXrtLaunchOps() {
  for (int i = 0; i < launch_nodes_.size(); ++i) {
    const XrtNode *node = launch_nodes_[i];
    // Add xrt launch operator
    OperatorConf op_conf;
    op_conf.set_name(node->name());
    DeviceType device_type = XrtDeviceToDeviceType(node->device());
    op_conf.set_device_type(device_type);

    XrtLaunchOpConf *launch_conf = op_conf.mutable_xrt_launch_conf();
    AddInBlobNames(node->in_edges(), launch_conf);
    AddOutBlobNames(node->out_edges(), launch_conf);

    buildXrtLaunchAttribute(node->sub_graph(), launch_conf->mutable_attr());

    if (IsAfterAllReduce(node) && node->out_edges().size() == 0) {
      launch_conf->mutable_attr()->set_model_update(true);
    }

    CHECK_GT(folded_nodes_[i].size(), 0);
    const ParallelConf &parallel_conf =
        builder_->ParallelConf4OpName(folded_nodes_[i][0]->name());
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

  auto AddControlInOpName = [&](OperatorConf *conf,
                                const std::string &op_name) -> void {
    std::string ctrl_in_op_name = op_name;
    const auto &it = folded_op_names.find(op_name);
    if (it != folded_op_names.end()) {
      ctrl_in_op_name = it->second->name();
    }
    if (conf->name() != ctrl_in_op_name) {
      DoNoDuplicationAdd(conf->mutable_ctrl_in_op_name(), ctrl_in_op_name);
    }
  };

  for (const XrtNode *node : graph_.Nodes()) {
    auto *op_conf = builder_->MutableOpConf4OpName(node->name());
    if (node->sub_graph() == nullptr) {
      auto ctrl_in_op_names = op_conf->ctrl_in_op_name();
      op_conf->clear_ctrl_in_op_name();
      for (const auto &op_name : ctrl_in_op_names) {
        AddControlInOpName(op_conf, op_name);
      }
    } else {
      for (const XrtNode *sub_node : node->sub_graph()->Nodes()) {
        if (sub_node->IsArgumentNode()) {
          continue;
        }
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
    auto *launch_conf = builder_->MutableOpConf4OpName(launch_op_name)
                            ->mutable_xrt_launch_conf();
    util::Map<std::string, std::string> fixed_blob_names;
    // Fix output blob names
    for (int i = 0; i < launch_conf->out().size(); ++i) {
      std::string blob_name = launch_conf->out()[i];
      std::string fixed_blob_name = absl::StrCat("out_", i);
      fixed_blob_names.emplace(blob_name, fixed_blob_name);

      launch_conf->mutable_out()->Mutable(i)->assign(fixed_blob_name);
      // Append to `batch_axis`
      if (builder_->HasBatchAxis4Lbn(blob_name)) {
        fixed_blob_name = absl::StrCat(launch_op_name, "/", fixed_blob_name);
        builder_->AddBatchAxis4Lbn(fixed_blob_name,
                                   builder_->BatchAxis4Lbn(blob_name));
      }
    }

    // Infect changes to the next operators only once
    std::unordered_set<const XrtNode *> changed_nodes;

    for (const XrtEdge *edge : node->out_edges()) {
      const XrtNode *end = edge->end();
      if (edge->IsControlEdge() || !changed_nodes.insert(end).second) {
        continue;
      }
      if (end->type() == _XrtLaunchOpType) {
        auto *launch_conf = builder_->MutableOpConf4OpName(end->name())
                                ->mutable_xrt_launch_conf();
        for (auto &blob_name : *launch_conf->mutable_in()) {
          const auto &it = fixed_blob_names.find(blob_name);
          if (it != fixed_blob_names.end()) {
            std::string fixed_blob_name =
                absl::StrCat(launch_op_name, "/", it->second);
            blob_name = fixed_blob_name;
          }
        }
        // Fix input argument blob name for subgraph first
        FixSubgraphInArgumentsBlobNames(end->sub_graph(), launch_op_name,
                                        launch_conf, fixed_blob_names);
      } else {
        // TODO(hjchen2) Current implementation is ugly
        auto *op_conf = builder_->MutableOpConf4OpName(end->name());
        DeviceType device_type = xrt::XrtDeviceToDeviceType(end->device());
        std::shared_ptr<Operator> op =
            ConstructOp(*op_conf, device_type, &GlobalJobDesc());
        for (const std::string &input : op->input_bns()) {
          const LogicalBlobId &lbi = op->BnInOp2Lbi(input);
          std::string blob_name = BlobIdToName(lbi);
          const auto &it = fixed_blob_names.find(blob_name);
          if (it != fixed_blob_names.end()) {
            std::string fixed_blob_name =
                absl::StrCat(launch_op_name, "/", it->second);
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
    const OpTimeShape &time_shape =
        builder_->TimeShape4OpName(folded_nodes_[i][0]->name());
    // TODO(hjchen2) check time shape for all folded nodes
    builder_->AddTimeShape4OpName(launch_nodes_[i]->name(), time_shape);
  }
}

void FoldSubgraphBuilder::FixupSbpSignatures() {
  for (const XrtNode *node : launch_nodes_) {
    OperatorConf *op_conf = builder_->MutableOpConf4OpName(node->name());
    std::shared_ptr<Operator> op = ConstructOp(*op_conf, &GlobalJobDesc());
    util::Map<std::string, std::string> blob_names;
    for (const std::string &bn : op->input_bns()) {
      std::string blob_name = BlobIdToName(op->BnInOp2Lbi(bn));
      blob_names.emplace(blob_name, bn);
    }
    for (const std::string &bn : op->output_bns()) {
      std::string blob_name = op->BnInOp2Lbi(bn).blob_name();
      blob_names.emplace(blob_name, bn);
    }

    SbpSignature sbp_conf;
    auto *sbp_signatures = sbp_conf.mutable_bn_in_op2sbp_parallel();
    for (const XrtEdge *edge : node->in_edges()) {
      CHECK(edge->HasAttr("sbp_policy"));
      std::string bn = blob_names.at(edge->argument().name());
      (*sbp_signatures)[bn] =
          edge->Attr<std::vector<SbpParallel>>("sbp_policy")[1];
    }
    for (const XrtEdge *edge : node->out_edges()) {
      CHECK(edge->HasAttr("sbp_policy"));
      std::string bn = blob_names.at(edge->argument().name());
      (*sbp_signatures)[bn] =
          edge->Attr<std::vector<SbpParallel>>("sbp_policy")[0];
    }
    // Append sbp signatures to helper
    builder_->AddSbpSignature4OpName(node->name(), sbp_conf);
  }
}

void FoldSubgraphBuilder::RemoveLaunchFoldedOps() {
  for (const XrtNode *node : launch_nodes_) {
    for (const XrtNode *sub_node : node->sub_graph()->Nodes()) {
      if (!sub_node->IsArgumentNode()) {
        builder_->RemoveOpByName(sub_node->name());
      }
    }
  }
}

void FoldSubgraphBuilder::InferIsAfterAllReduce() {
  algorithm::TopologyVisit(graph_, [this](const XrtNode *node) {
    for (const XrtEdge *edge : node->in_edges()) {
      const XrtNode *start = edge->start();
      if (IsAfterAllReduce(start) || start->type() == _ReduceSplitType) {
        after_allreduce_nodes_.insert(node);
      }
    }
  });
}

bool FoldSubgraphBuilder::IsAfterAllReduce(const XrtNode *node) {
  return after_allreduce_nodes_.count(node) > 0;
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
    CHECK_GE(params.size(), 1)
        << "Job is required by `RebuildCompiledJobPass`.";
    auto *job = any_cast<Job *>(params[0]);

    CHECK(graph) << "Graph is required by `RebuildCompiledJobPass`.";
    FoldSubgraphBuilder(*graph, job).Build();
  }
};

REGISTER_XRT_PASS(RebuildCompiledJob, RebuildCompiledJobPass);

}  // namespace xrt
}  // namespace oneflow
