#include "operator/relu_op.h"
#include "glog/logging.h"

namespace oneflow {

namespace {

void InitDataBlobNameSet(DataBlobNameSet& cur_set) {
  cur_set.input_blob_names.push_back("in");
  cur_set.input_diff_blob_names.push_back("in_diff");
  cur_set.output_blob_names.push_back("out");
  cur_set.output_diff_blob_names.push_back("out_diff");
}

void InitModelBlobNameSet(ModelBlobNameSet& cur_set) {
}

}

void ReluOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_relu_op_conf());
  auto cnf_ptr = new ReluOpConf(op_conf.relu_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  InitDataBlobNameSet(mutable_data_blob_name_set());
  InitModelBlobNameSet(mutable_model_blob_name_set());
}

} // namespace oneflow
