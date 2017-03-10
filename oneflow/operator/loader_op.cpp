#include "operator/loader_op.h"
#include "glog/logging.h"

namespace oneflow {

namespace {

void InitDataBlobNameSet(DataBlobNameSet& cur_set) {
  cur_set.output_blob_names.push_back("data");
  cur_set.output_blob_names.push_back("label");
}

void InitModelBlobNameSet(ModelBlobNameSet& cur_set) {
}

}

void LoaderOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_loader_op_conf());
  auto cnf_ptr = new LoaderOpConf(op_conf.loader_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  InitDataBlobNameSet(mutable_data_blob_name_set());
  InitModelBlobNameSet(mutable_model_blob_name_set());
}

} // namespace oneflow
