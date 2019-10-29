#include <string>
#include <vector>
#include "glog/logging.h"

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

namespace {

ParallelContext BuildLocalParallelContext() {
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(0);
  parallel_ctx.set_parallel_num(1);
  return std::move(parallel_ctx);
}

SbpSignature BuildLocalSbpSignature(const std::shared_ptr<Operator> &op) {
  SbpSignature sbp_signature;
  auto &sbp_detail = *(sbp_signature.mutable_bn_in_op2sbp_parallel());
  for (const auto &name : op->input_bns()) {
    sbp_detail[name].mutable_split_parallel()->set_axis(0);
  }
  for (const auto &name : op->output_bns()) {
    sbp_detail[name].mutable_split_parallel()->set_axis(0);
  }
  return std::move(sbp_signature);
}

void InferShape(XrtGraph *graph, const XrtPassOptions &options,
                const JobDesc *job_desc,
                util::Map<std::string, BlobDesc> *blob_descs) {
  algorithm::TopologyVisit(*graph, [&](XrtNode *node) {
    if (node->IsArgumentNode()) {
      const auto &conf =
          *dynamic_cast<const XrtLaunchOpConf::Argument *>(&node->param());
      blob_descs->emplace(conf.out(), blob_descs->at(conf.in()));
    } else {
      DeviceType device_type = BackendToDeviceType(node->backend());
      const auto &conf = *dynamic_cast<const OperatorConf *>(&node->param());
      auto op = ConstructOp(conf, device_type, job_desc);
      auto get_blob_desc_fn = [&](const std::string &bn) -> BlobDesc * {
        const LogicalBlobId &lbi = op->BnInOp2Lbi(bn);
        std::string blob_name = BlobIdToName(lbi);
        return &((*blob_descs)[blob_name]);
      };
      auto parallel_ctx = BuildLocalParallelContext();
      auto sbp_signature = BuildLocalSbpSignature(op);
      op->InferBlobDescsIf(get_blob_desc_fn, &parallel_ctx, &sbp_signature,
                           [](OpContext *) {});
    }
    // Update blob desc on the output edges
    for (XrtEdge *edge : node->out_edges()) {
      std::string name = edge->argument().name();
      auto it = blob_descs->find(name);
      CHECK(it != blob_descs->end());
      Argument argument(name, it->second.shape(), it->second.data_type());
      edge->SetArgument(argument);
    }
  });
}

}  // namespace

class InferShapePass : public XrtPass {
 public:
  InferShapePass() = default;

  void Run(XrtGraph *graph, const XrtPassOptions &options,
           const std::vector<Any> &params) override {
    CHECK_GE(params.size(), 2) << "JobDesc and BlobDesc are required.";
    const JobDesc *job_desc = any_cast<const JobDesc *>(params[0]);
    auto *blob_descs = any_cast<util::Map<std::string, BlobDesc> *>(params[1]);
    InferShape(graph, options, job_desc, blob_descs);
  }
};

REGISTER_XRT_PASS(InferShape, InferShapePass);

}  // namespace xrt
}  // namespace oneflow
