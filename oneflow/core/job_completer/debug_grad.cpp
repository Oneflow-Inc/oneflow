#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_debug_conf());
  const DebugOpConf& conf = op.op_conf().debug_conf();
  if (conf.has_out_diff_blob_dump_dir()
      || conf.const_in_diff_case() != DebugOpConf::CONST_IN_DIFF_NOT_SET) {
    op_confs->push_back(OperatorConf());
    OperatorConf* diff_op_conf = &op_confs->back();
    diff_op_conf->set_name(op.op_name() + "_grad");
    DebugOpConf* diff_conf = diff_op_conf->mutable_debug_conf();
    diff_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    diff_conf->set_out("out");
    if (conf.has_out_diff_blob_dump_dir()) {
      diff_conf->set_in_blob_dump_dir(conf.out_diff_blob_dump_dir());
    }
    if (conf.has_part_name_prefix()) { diff_conf->set_part_name_prefix(conf.part_name_prefix()); }
    if (conf.has_part_name_suffix_length()) {
      diff_conf->set_part_name_suffix_length(conf.part_name_suffix_length());
    }
    if (conf.has_const_in_diff_feature_load_filepath()) {
      diff_conf->set_const_out_feature_load_filepath(conf.const_in_diff_feature_load_filepath());
    } else if (conf.has_const_in_diff_feature()) {
      *diff_conf->mutable_const_out_feature() = conf.const_in_diff_feature();
    } else {
      // do nothing
    }
    DiffLbi4BnInOp("in")->set_op_name(diff_op_conf->name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
  } else {
    *DiffLbi4BnInOp("in") = *DiffLbi4BnInOp("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kDebugConf, &GenerateBackwardOpConf);

}  // namespace oneflow
