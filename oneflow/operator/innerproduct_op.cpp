#include "operator/innerproduct_op.h"
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
  cur_set.model_blob_names.push_back("weight");
  cur_set.model_diff_blob_names.push_back("weight_diff");
  cur_set.model_blob_names.push_back("bias");
  cur_set.model_diff_blob_names.push_back("bias_diff");
  cur_set.model_tmp_blob_names.push_back("bias_multiplier");
}

}

void InnerProductOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_inner_product_op_conf());
  auto cnf_ptr =
      new InnerProductOpConf(op_conf.inner_product_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  InitDataBlobNameSet(mutable_data_blob_name_set());
  InitModelBlobNameSet(mutable_model_blob_name_set());
}

} // namespace oneflow
