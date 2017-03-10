#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"

namespace oneflow {

namespace {

void InitDataBlobNameSet(DataBlobNameSet& cur_set) {
  cur_set.input_blob_names.push_back("data");
  cur_set.input_diff_blob_names.push_back("data_diff");
  cur_set.input_blob_names.push_back("label");
  cur_set.input_diff_blob_names.push_back("label_diff");
  cur_set.output_blob_names.push_back("loss");
  cur_set.data_tmp_blob_names.push_back("loss_buffer");
}

void InitModelBlobNameSet(ModelBlobNameSet& cur_set) {
}

}

void MultinomialLogisticLossOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_multinomial_logistic_loss_op_conf());
  auto cnf_ptr =
      new MultinomialLogisticLossOpConf(
          op_conf.multinomial_logistic_loss_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  InitDataBlobNameSet(mutable_data_blob_name_set());
  InitModelBlobNameSet(mutable_model_blob_name_set());
}

} // namespace oneflow
