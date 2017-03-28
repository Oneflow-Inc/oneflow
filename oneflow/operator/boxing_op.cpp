#include "operator/boxing_op.h"

namespace oneflow {

void BoxingOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();

  CHECK(op_conf.has_boxing_op_conf());
  auto cnf = new BoxingOpConf(op_conf.boxing_op_conf());
  mut_pb_op_conf().reset(cnf);

  int32_t in_num = 0;
  if (cnf->in_box_conf_case() == BoxingOpConf::kConcatInBoxConf) {
    CHECK(cnf->has_concat_in_box_conf());
    in_num = cnf->concat_in_box_conf().proportion_size();
  } else {
    CHECK(cnf->has_add_in_box_conf());
    in_num = cnf->add_in_box_conf().in_num();
  }
  int32_t out_num = 0;
  if (cnf->out_box_conf_case() == BoxingOpConf::kSplitOutBoxConf) {
    CHECK(cnf->has_split_out_box_conf());
    out_num = cnf->split_out_box_conf().proportion_size();
  } else {
    CHECK(cnf->has_clone_out_box_conf());
    out_num = cnf->clone_out_box_conf().out_num();
  }
  for (int32_t i = 0; i < in_num; ++i) {
    RegisterInputBlobName(cnf->lbn() + "/in_" + std::to_string(i));
  }
  for (int32_t i = 0; i < out_num; ++i) {
    RegisterOutputBlobName(cnf->lbn() + "/out_" + std::to_string(i));
  }
  RegisterDataTmpBlobName("middle");
}

std::string BoxingOp::ibn2lbn(const std::string& input_blob_name) const {
  size_t slash_pos = input_blob_name.rfind('/');
  return input_blob_name.substr(0, slash_pos);
}

std::string BoxingOp::obn2lbn(const std::string& output_blob_name) const {
  size_t slash_pos = output_blob_name.rfind('/');
  return output_blob_name.substr(0, slash_pos);
}

std::string BoxingOp::idbn2lbn(const std::string& input_diff_blob_name) const {
  LOG(FATAL) << "invalid";
  return "";
}

std::string BoxingOp::odbn2lbn(const std::string& output_diff_blob_name) const {
  LOG(FATAL) << "invalid";
  return "";
}

} // namespace oneflow
