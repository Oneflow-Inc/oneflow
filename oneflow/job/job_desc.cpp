#include "job/job_desc.h"
#include "common/proto_io.h"

namespace oneflow {

void JobDesc::InitFromJobConf(const JobConf& conf) {
  ParseProtoFromTextFile(conf.train_dlnet_conf_filepath(), &train_dl_net_conf_);
  ParseProtoFromTextFile(conf.resource_filepath(), &resource_);
  ParseProtoFromTextFile(conf.strategy_filepath(), &strategy_);
  md_load_machine_ = conf.model_load_machine();
  md_save_machine_ = conf.model_save_machine();
}

void JobDesc::InitFromProto(const JobDescProto&) {
  TODO();
}

JobDescProto JobDesc::ToProto() const {
  TODO();
}

} // namespace oneflow
