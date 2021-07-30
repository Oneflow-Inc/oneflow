#ifndef ONEFLOW_API_JAVA_ENV_JOB_API_H_
#define ONEFLOW_API_JAVA_ENV_JOB_API_H_

#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include "oneflow/api/python/framework/framework.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_instance.h"

namespace oneflow {

class JavaForeignJobInstance : public JobInstance {
 public:
  JavaForeignJobInstance(std::string job_name,  std::string sole_input_op_name_in_user_job,
                         std::string sole_output_op_name_in_user_job, std::function<void(uint64_t)> push_cb,
                         std::function<void(uint64_t)> pull_cb, std::function<void()> finish) : 
                           job_name_(job_name), 
                           sole_input_op_name_in_user_job_(sole_input_op_name_in_user_job),
                           sole_output_op_name_in_user_job_(sole_output_op_name_in_user_job),
                           push_cb_(push_cb),
                           pull_cb_(pull_cb),
                           finish_(finish) {
  }
  ~JavaForeignJobInstance() {}
  std::string job_name() const { return job_name_; }
  std::string sole_input_op_name_in_user_job() const { return sole_input_op_name_in_user_job_; }
  std::string sole_output_op_name_in_user_job() const { return sole_output_op_name_in_user_job_; }
  void PushBlob(uint64_t ofblob_ptr) const {
    if (push_cb_ != nullptr) push_cb_(ofblob_ptr);
  }
  void PullBlob(uint64_t ofblob_ptr) const {
    if (pull_cb_ != nullptr) pull_cb_(ofblob_ptr);
  }
  void Finish() const {
    if (finish_ != nullptr) finish_();
  }

 private:
  std::string job_name_;
  std::string sole_input_op_name_in_user_job_;
  std::string sole_output_op_name_in_user_job_;
  std::function<void(uint64_t)> push_cb_;
  std::function<void(uint64_t)> pull_cb_;
  std::function<void()> finish_;
};

}

struct PullTensor {
  unsigned char* data_;
  uint64_t len_;
  int dtype_;
  long* shape_;
  size_t axes_;

  ~PullTensor() {
    if (data_) {
      delete[] data_;
      data_ = nullptr;
    }

    if (shape_) {
      delete[] shape_;
      shape_ = nullptr;
    }
  }
};

inline void SetJobConfForCurJobBuildAndInferCtx(const std::string& job_conf_proto) {
  oneflow::JobConfigProto job_conf;
  oneflow::TxtString2PbMessage(job_conf_proto, &job_conf);
  oneflow::cfg::JobConfigProto job_conf_cfg;
  job_conf_cfg.InitFromProto(job_conf);

  // if (batch_size > 0) {
  //   oneflow::cfg::JobSignatureDef* signature = job_conf_cfg.mutable_signature();
  //   if (signature != nullptr) {
  //     for (auto iter = signature->mutable_inputs()->begin(); iter != signature->mutable_inputs()->end(); iter++) {
  //       iter->second.mutable_blob_conf()->mutable_shape()->set_dim(0, batch_size);
  //     } 
  //   }
  // }
  // std::cout << job_conf_cfg.DebugString() << std::endl;

  oneflow::CurJobBuildAndInferCtx_SetJobConf(job_conf_cfg).GetOrThrow();
}

inline void SetScopeForCurJob(const std::string& job_conf_proto,
                              const std::string& ids,
                              const std::string& device) {
  // Todo: user configuration
  oneflow::JobConfigProto job_conf;
  oneflow::TxtString2PbMessage(job_conf_proto, &job_conf);

  std::shared_ptr<oneflow::cfg::JobConfigProto> job_conf_cfg = std::make_shared<oneflow::cfg::JobConfigProto>();
  job_conf_cfg->InitFromProto(job_conf);

  std::shared_ptr<oneflow::Scope> scope;
  auto BuildInitialScope = [&scope, &job_conf_cfg, &ids, &device](oneflow::InstructionsBuilder* builder) mutable -> oneflow::Maybe<void> {
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    const std::vector<std::string> machine_device_ids({ ids });
    std::shared_ptr<oneflow::Scope> initialScope = builder->BuildInitialScope(session_id, job_conf_cfg, device, machine_device_ids, nullptr, false).GetPtrOrThrow();
    scope = initialScope;
    return oneflow::Maybe<void>::Ok();
  };
  oneflow::LogicalRun(BuildInitialScope);
  oneflow::ThreadLocalScopeStackPush(scope).GetOrThrow();  // fixme: bug?
}

inline void CurJobAddOp(const std::string& op_conf_proto) {
  oneflow::OperatorConf op_conf;
  oneflow::TxtString2PbMessage(op_conf_proto, &op_conf);
  auto scope = oneflow::GetCurrentScope().GetPtrOrThrow();
  op_conf.set_scope_symbol_id(scope->symbol_id().GetOrThrow());
  op_conf.set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  oneflow::CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf.DebugString());
}

inline void LoadCheckPoint(const std::string& load_job_name, signed char* path, int64_t path_length) {
  int64_t *shape = new int64_t[1]{ path_length };  // Todo: is there a better way to allocate memory?

  auto copy_model_load_path = [shape, path, path_length](uint64_t of_blob_ptr) -> void {
    using namespace oneflow;
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->CopyShapeFrom(shape, 1);
    of_blob->AutoMemCopyFrom(path, path_length);

    delete []shape;
  };
  const std::shared_ptr<oneflow::JobInstance> job_inst(
    new oneflow::JavaForeignJobInstance(load_job_name, "", "", copy_model_load_path, nullptr, nullptr)
  );
  oneflow::LaunchJob(job_inst);
}

inline void RunPushJob(const std::string& job_name,
                             const std::string& op_name,
                             void* data,
                             int dtype_code,
                             int64_t* shape,
                             int64_t shape_length) {
  int64_t element_number = 1;
  for (int64_t i = 0; i < shape_length; i++) {
    element_number = element_number * shape[i];
  }

  auto job_instance_fun = [=](uint64_t of_blob_ptr) -> void {
    using namespace oneflow;
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->CopyShapeFrom(shape, shape_length);

    // Todo: support more data type
    if (dtype_code == kFloat) {
      of_blob->AutoMemCopyFrom((float*) data, element_number);
    }
    if (dtype_code == kInt32) {
      of_blob->AutoMemCopyFrom((int*) data, element_number);
    }
  };
  const std::shared_ptr<oneflow::JobInstance> job_instance(
    new oneflow::JavaForeignJobInstance(job_name, op_name, "", job_instance_fun, nullptr, nullptr)
  );
  oneflow::LaunchJob(job_instance);
}

inline void RunJob(const std::string& job_name) {
  const std::shared_ptr<oneflow::JobInstance> job_inst(
    new oneflow::JavaForeignJobInstance(job_name, "", "", nullptr, nullptr, nullptr)
  );
  oneflow::LaunchJob(job_inst);
}

inline void RunPullJobSync(const std::string& job_name,
                           const std::string& op_name,
                           std::shared_ptr<PullTensor>& pull_tensor) {
  std::promise<std::shared_ptr<PullTensor>> tensor_promise;
  std::future<std::shared_ptr<PullTensor>> tensor_future = tensor_promise.get_future();

  auto job_instance_fun = [&](uint64_t of_blob_ptr) -> void {
    using namespace oneflow;
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    std::shared_ptr<PullTensor> tensor = std::make_shared<PullTensor>();
    
    // shape
    tensor->axes_ = of_blob->NumAxes();
    tensor->shape_ = new int64_t[tensor->axes_];
    of_blob->CopyShapeTo(tensor->shape_, tensor->axes_);

    // data
    int64_t element_number = 1;
    for (int i = 0; i < tensor->axes_; i++) {
      element_number = element_number * tensor->shape_[i];
    }

    // Todo: support more data type
    if (of_blob->data_type() == kFloat) {
      element_number = element_number * 4;
    }
    if (of_blob->data_type() == kInt32) {
      element_number = element_number * 4;
    }
    tensor->dtype_ = of_blob->data_type();
    tensor->len_ = element_number;
    tensor->data_ = new unsigned char[element_number];

    if (of_blob->data_type() == kFloat) {
      of_blob->AutoMemCopyTo((float*) tensor->data_, element_number / 4);
    }
    if (of_blob->data_type() == kInt32) {
      of_blob->AutoMemCopyTo((int*) tensor->data_, element_number / 4);
    }

    tensor_promise.set_value(tensor);
  };

  const std::shared_ptr<oneflow::JobInstance> job_instance(
    new oneflow::JavaForeignJobInstance(job_name, "", op_name, nullptr, job_instance_fun, nullptr)
  );
  oneflow::LaunchJob(job_instance);

  // we need to wait for the result since LaunchJob is asynchronous
  pull_tensor = tensor_future.get();
}

#endif  // ONEFLOW_API_JAVA_ENV_JOB_API_H_
