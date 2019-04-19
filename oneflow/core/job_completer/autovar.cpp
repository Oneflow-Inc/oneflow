#include "oneflow/core/job_completer/autovar.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

OperatorConf GenerateVariableOpConf(const BlobDesc& blob_desc, const std::string& name,
                                    const std::string& model_name) {
  OperatorConf var_op;
  var_op.set_name(name);
  VariableOpConf* var_op_conf = var_op.mutable_variable_conf();
  var_op_conf->set_out("out");
  blob_desc.shape().ToProto(var_op_conf->mutable_shape());
  var_op_conf->set_data_type(blob_desc.data_type());
  var_op_conf->set_model_name(model_name);
  return var_op;
}

void GenerateInputVarOpConfIf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  if (IsClassRegistered<GenerateInputVarOpConfWrapperStruct>(op.op_conf().op_type_case())) {
    auto* obj = NewObj<GenerateInputVarOpConfWrapperStruct>(op.op_conf().op_type_case());
    obj->Call(op, op_confs, LogicalBlobDesc4BnInOp);
  }
}

void AutoVar(const OpGraph& op_graph, Job* job) {
  HashMap<LogicalBlobId, BlobDesc> lbi2unparalleled_blob_desc;
  op_graph.TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    auto MutUnparalleledBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
      return &lbi2unparalleled_blob_desc[op_node->op().BnInOp2Lbi(bn)];
    };
    // the real important data we want to get is:
    // a) model blobs' byte size;
    // b) number of axes of blobs' body shape;
    // Hence the argument record_piece_size can be any positive number, here it's 1
    op_node->op().InferBlobDescsIf(MutUnparalleledBlobDesc4BnInOp, &parallel_ctx, 1,
                                   [](OpContext*) {});
  });
  JobBuilder job_builder(job);

  op_graph.ForEachNode([&](OpNode* op_node) {
    std::vector<OperatorConf> ops;
    auto UnparalleledBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return lbi2unparalleled_blob_desc.at(op_node->op().BnInOp2Lbi(bn));
    };
    GenerateInputVarOpConfIf(op_node->op(), &ops, UnparalleledBlobDesc4BnInOp);
    if (!ops.empty()) { job_builder.AddOrMutOps(op_node->parallel_desc().parallel_conf(), ops); }
  });
}

}  // namespace oneflow
