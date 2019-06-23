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

void DoNoDuplicationAdd(XlaArgumentOpConf::DataVec *data_vec,
                        const std::string &data) {
  const auto &d = data_vec->data();
  if (std::find(d.begin(), d.end(), data) == d.end()) {
    data_vec->add_data(data);
  }
}

void DoNoDuplicationAddInput(XlaLaunchOpConf *launch_conf,
                             const std::string &in) {
  const auto &d = launch_conf->in();
  if (std::find(d.begin(), d.end(), in) == d.end()) {
    launch_conf->add_in(in);
  }
}

void DoNoDuplicationAddOutput(XlaLaunchOpConf *launch_conf,
                              const std::string &out) {
  const auto &d = launch_conf->out();
  if (std::find(d.begin(), d.end(), out) == d.end()) {
    launch_conf->add_out(out);
  }
}

std::string FixBlobNamePrefix(const std::string &blob_name,
                              const std::string &prefix) {
  std::vector<std::string> name_split = absl::StrSplit(blob_name, "/");
  CHECK_EQ(name_split.size(), 2);
  return absl::StrCat(prefix, "/", name_split[1]);
}

XlaLaunchOpConf *MutableXlaLaunchOpConf(Job *job, const std::string &op_name) {
  for (auto &conf : *(job->mutable_net()->mutable_op())) {
    if (conf.name() == op_name) {
      return conf.mutable_xla_launch_conf();
    }
  }
  return nullptr;
}

XlaArgumentOpConf *MutableXlaArgumentOpConf(XlaLaunchOpConf *launch_conf,
                                            const std::string &op_name) {
  for (auto &conf : *(launch_conf->mutable_attr()->mutable_node())) {
    if (conf.name() == op_name) {
      return conf.mutable_xla_argument_conf();
    }
  }
  return nullptr;
}

void FixSubgraphInArgumentsBlobNames(const mola::XlaGraph *graph,
                                     XlaLaunchOpConf *launch_conf,
                                     const std::string &blob_name,
                                     const std::string &fixed_blob_name) {
  for (const auto *node : graph->Nodes()) {
    if (absl::StartsWith(node->op_name(), mola::_XlaInArgumentPrefix)) {
      auto *argument_conf = MutableXlaArgumentOpConf(launch_conf,
                                                     node->op_name());
      for (auto &argument_blob_name :
           *(argument_conf->mutable_out()->mutable_data())) {
        if (argument_blob_name == blob_name) {
          argument_blob_name = fixed_blob_name;
        }
      }
    }
  }
}

void FixSubgraphOutArgumentsBlobNames(const mola::XlaGraph *graph,
                                      XlaLaunchOpConf *launch_conf,
                                      const std::string &launch_op_name) {
  for (const auto *node : graph->Nodes()) {
    if (absl::StartsWith(node->op_name(), mola::_XlaOutArgumentPrefix)) {
      auto *argument_conf = MutableXlaArgumentOpConf(launch_conf,
                                                     node->op_name());
      for (auto &blob_name :
           *(argument_conf->mutable_in()->mutable_data())) {
        blob_name = FixBlobNamePrefix(blob_name, launch_op_name);
      }
    }
  }
}

void FixXlaLaunchBlobNames(const mola::XlaGraph &graph, Job *job,
                           JobBuilder *builder) {
  for (const XlaNode *node : graph.Nodes()) {
    if (node->op_type() != mola::_XlaLaunchOpType) {
      continue;
    }
    std::string launch_op_name = node->op_name();
    XlaLaunchOpConf *launch_conf = MutableXlaLaunchOpConf(job, launch_op_name);
    for (auto &blob_name : *launch_conf->mutable_out()) {
      // Set fixed output blob name
      blob_name = FixBlobNamePrefix(blob_name, launch_op_name);
    }

    for (const XlaEdge *edge : node->out_edges()) {
      if (!edge->IsControlEdge()) {
        const XlaNode *end = edge->end();
        const Argument &arg = edge->argument();

        if (end->op_type() == mola::_XlaLaunchOpType) {
          auto *launch_conf = MutableXlaLaunchOpConf(job, end->op_name());
          for (auto &blob_name : *launch_conf->mutable_in()) {
            if (blob_name == arg.blob_name()) {
              std::string fixed_blob_name =
                  FixBlobNamePrefix(blob_name, launch_op_name);
              // Fix input argument blob name for subgraph
              FixSubgraphInArgumentsBlobNames(
                  end->sub_graph(), launch_conf, blob_name, fixed_blob_name);
              blob_name = fixed_blob_name;
            }
          }
        } else {
          // TODO(hjchen2) This implementation is ugly
          PbMessage *op_conf = const_cast<PbMessage *>(&(end->proto_conf()));
          for (const std::string &input : end->op()->input_bns()) {
            const LogicalBlobId &lbi = end->op()->BnInOp2Lbi(input);
            if (absl::StrCat(lbi.op_name(), "/", lbi.blob_name()) ==
                arg.blob_name()) {
              std::string fixed_blob_name =
                  FixBlobNamePrefix(arg.blob_name(), launch_op_name);
              // Fix input blob name for normal node
              SetBnValInOpTypeConf(op_conf, input, arg.blob_name(),
                                   fixed_blob_name);
            }
          }
          builder->MutOps({end->op()->op_conf()});
        }
      }
    }
    // Subgraph output argument blob name
    FixSubgraphOutArgumentsBlobNames(
        node->sub_graph(), launch_conf, launch_op_name);
  }
}

void buildXlaLaunchAttribute(JobBuilder *builder,
                             const mola::XlaGraph *graph,
                             XlaLaunchOpConf::Attribute *launch_attr) {
  for (const XlaNode *node : graph->Nodes()) {
    if (node->op_type() != mola::_XlaArgumentOpType) {
      builder->RemoveOp(node->op()->op_name());
      *(launch_attr->add_node()) = node->op()->op_conf();
    } else {
      OperatorConf op_conf;
      op_conf.set_name(node->op_name());
      XlaArgumentOpConf *argument_conf = op_conf.mutable_xla_argument_conf();
      // Usually one argument node has either inputs or outputs
      CHECK(node->in_edges().size() == 0 || node->out_edges().size() == 0);
      // Build inputs or outputs for the argument nodes
      if (node->in_edges().size() > 0) {
        auto *in_vec = argument_conf->mutable_in(); 
        for (const XlaEdge *edge : node->in_edges()) {
          const Argument &argument = edge->argument();
          DoNoDuplicationAdd(in_vec, argument.blob_name());
        }
      }
      if (node->out_edges().size() > 0) {
        auto *out_vec = argument_conf->mutable_out();
        for (const XlaEdge *edge : node->out_edges()) {
          const Argument &argument = edge->argument();
          DoNoDuplicationAdd(out_vec, argument.blob_name());
        }
      }
      *(launch_attr->add_node()) = op_conf;
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
    OperatorConf launch_op_conf; 
    launch_op_conf.set_name(node->op_name());

    XlaLaunchOpConf *launch_conf = launch_op_conf.mutable_xla_launch_conf();
    for (const XlaEdge *edge : node->in_edges()) {
      if (!edge->IsControlEdge()) {
        const Argument &arg = edge->argument();
        DoNoDuplicationAddInput(launch_conf, arg.blob_name());
      }
    }
    for (const XlaEdge *edge : node->out_edges()) {
      if (!edge->IsControlEdge()) {
        const Argument &arg = edge->argument();
        DoNoDuplicationAddOutput(launch_conf, arg.blob_name());
      }
    }

    XlaLaunchOpConf::Attribute *launch_attr = launch_conf->mutable_attr();
    buildXlaLaunchAttribute(&builder, node->sub_graph(), launch_attr);
   
    // TODO(hjchen2) Assign parallel conf
    ParallelConf parallel_conf;
    builder.AddOps(parallel_conf, {launch_op_conf});
  }

  // Fix blob names
  FixXlaLaunchBlobNames(graph, job, &builder);
}

}  // namespace oneflow
