/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/framework/config_def.h"

namespace oneflow {

namespace cfg {
class JobConfigProto;
}

bool IsInterfaceOpConf(const OperatorConf& op_conf);

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc(const JobConfigProto& job_conf, int64_t job_id);
  explicit JobDesc(const JobConfigProto& job_conf) : JobDesc(job_conf, -1) {}
  ~JobDesc() = default;

  static Maybe<JobDesc> New(int64_t symbol_id, const JobConfigProto& job_conf);
  const Maybe<int64_t>& symbol_id() const { return symbol_id_; }
  const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf() const { return cfg_job_conf_; }

  // Common
  int64_t job_id() const { return job_id_; }
  const std::string& job_name() const { return job_conf_.job_name(); }
  int64_t concurrency_width() const { return job_conf_.concurrency_width(); }
  const JobConfigProto& job_conf() const { return job_conf_; }
  DataType DefaultDataType() const { return job_conf_.default_data_type(); }
  bool EnableCudnn() const { return job_conf_.enable_cudnn(); }
  bool IsTrain() const { return job_conf_.has_train_conf(); }
  bool IsPredict() const { return job_conf_.has_predict_conf(); }
  int64_t piece_num_of_experiment_phase() const;
  bool use_memory_allocation_algorithm_v2() const {
    return job_conf_.use_memory_allocation_algorithm_v2();
  }
  bool enable_experiment_run() const;
  bool enable_reuse_mem() const { return job_conf_.enable_reuse_mem(); }
  bool enable_inplace() const { return job_conf_.enable_inplace(); }
  bool enable_auto_mixed_precision() const { return job_conf_.enable_auto_mixed_precision(); }
  bool do_parallel_cast_before_widening_type_cast() const {
    return job_conf_.do_parallel_cast_before_widening_type_cast();
  };
  bool prune_parallel_cast_ops() const { return job_conf_.prune_parallel_cast_ops(); }
  bool prune_cast_to_static_shape_ops() const { return job_conf_.prune_cast_to_static_shape_ops(); }
  int64_t cudnn_buf_limit_mbyte() const { return job_conf_.cudnn_buf_limit_mbyte(); }

  bool has_xrt_config() const { return job_conf_.has_xrt_config(); }
  const XrtConfig& xrt_config() const { return job_conf_.xrt_config(); }

#define DEFINE_FUNCTION_CONFIG_GETTER(T, func_name, field_name) \
  T func_name(const std::string& field_name) const {            \
    const AttrValue& attr_val = GetFunctionFlagVal(field_name); \
    CHECK(attr_val.has_##field_name());                         \
    return attr_val.field_name();                               \
  }
  DEFINE_FUNCTION_CONFIG_GETTER(bool, Bool, at_bool);
  DEFINE_FUNCTION_CONFIG_GETTER(int64_t, Int64, at_int64);
  DEFINE_FUNCTION_CONFIG_GETTER(double, Double, at_double);
  DEFINE_FUNCTION_CONFIG_GETTER(const std::string&, String, at_string);

  // Train conf
  int64_t TotalBatchNum() const;
  int64_t NumOfPiecesInBatch() const;

 private:
  Maybe<void> Init();
  const AttrValue& GetFunctionFlagVal(const std::string& field_name) const;

  JobConfigProto job_conf_;
  int64_t job_id_;
  Maybe<int64_t> symbol_id_;
  // merge job_conf_ and cfg_job_conf_ after cfg::JobConfigProto taken as a constructor argument
  std::shared_ptr<cfg::JobConfigProto> cfg_job_conf_;
};

typedef HashMap<std::string, int64_t> JobName2JobId;

class GlobalJobDescScope final {
 public:
  GlobalJobDescScope(const JobConfigProto& job_conf, int64_t job_id);
  ~GlobalJobDescScope();
};
const JobDesc& GlobalJobDesc();

bool IsPullJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info);
bool IsPushJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
