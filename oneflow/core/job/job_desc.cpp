#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void JobDesc::InitFromJobConf(const JobConf& conf) {
  LOG(INFO) << "Read JobConf";
  job_conf_ = conf;
  ParseProtoFromTextFile(conf.dlnet_filepath(), &train_dlnet_conf_);
  ParseProtoFromTextFile(conf.resource_filepath(), &resource_);
  ParseProtoFromTextFile(conf.placement_filepath(), &placement_);
}

void JobDesc::InitFromProto(const JobDescProto& proto) {
  LOG(INFO) << "Init JobDesc from Proto";
  job_conf_ = proto.job_conf();
  train_dlnet_conf_ = proto.train_dlnet_conf();
  resource_ = proto.resource();
  placement_ = proto.placement();
}

void JobDesc::ToProto(JobDescProto* proto) const {
  *(proto->mutable_job_conf()) = job_conf_;
  *(proto->mutable_train_dlnet_conf()) = train_dlnet_conf_;
  *(proto->mutable_resource()) = resource_;
  *(proto->mutable_placement()) = placement_;
}

}  // namespace oneflow
