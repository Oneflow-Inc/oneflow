#include "operator/convolution_op.h"
#include "glog/logging.h"

namespace oneflow {

namespace {

void InitDataBlobNameSet(DataBlobNameSet& cur_set) {
  cur_set.input_blob_names.push_back("in");
  cur_set.input_diff_blob_names.push_back("in_diff");
  cur_set.output_blob_names.push_back("out");
  cur_set.output_diff_blob_names.push_back("out_diff");
  cur_set.data_tmp_blob_names.push_back("col_buf");
}

void InitModelBlobNameSet(ModelBlobNameSet& cur_set) {
  cur_set.model_blob_names.push_back("weight");
  cur_set.model_diff_blob_names.push_back("weight_diff");
  cur_set.model_blob_names.push_back("bias");
  cur_set.model_diff_blob_names.push_back("bias_diff");
  cur_set.model_tmp_blob_names.push_back("bias_multiplier");
}

}

void ConvolutionOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_convolution_op_conf());
  auto cnf_ptr = new ConvolutionOpConf(op_conf.convolution_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  InitDataBlobNameSet(mutable_data_blob_name_set());
  InitModelBlobNameSet(mutable_model_blob_name_set());
}

} // namespace oneflow
