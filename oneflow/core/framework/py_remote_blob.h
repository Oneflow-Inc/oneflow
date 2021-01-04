#ifndef ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_
#define ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_


#include <typeinfo>
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/py_blob_desc.h"

namespace oneflow {

namespace compatible_py {

class ConsistentBlob: public BlobDesc {
 public:
  ConsistentBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, std::string job_name, const std::shared_ptr<Distribute>& distribute): BlobDesc(lbi. distribute), parallel_size_(0) {
    if (job_name.empty()) {
      std::shared_ptr<JobBuildAndInferCtxMgr> mgr;
      if (EagerExecutionEnabled()) {
        mgr =  JUST(GlobalMaybe<EagerJobBuildAndInferCtxMgr>());
      } else {
        mgr = JUST(GlobalMaybe<LazyJobBuildAndInferCtxMgr>());
      }
      job_name =  mgr->GetCurrentJobName();
    }
    job_name_ = job_name;
  }
  ConsistentBlob(const ConsistentBlob& consistent_blob) = default;
  ~ConsistentBlob = default;

  virtual std::shared_ptr<ConsistentBlob> with_distribute(
      const std::shared_ptr<Distribute>& distribute) const override {
    std::shared_ptr<BlobDesc> ret = Clone();
    ret.set_distribute(distribute);
    return ret;
  }

  int64_t parallel_size() {
    if (parallel_size_ == 0) {
      std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
      ParallelConf proto_parallel_conf;
      cfg_parallel_conf->ToProto(&proto_parallel_conf);
      ParallelDesc parallel_desc(proto_parallel_conf);
      parallel_size_ = parallel_desc.parallel_num();
    }
    return parallel_size_;
  }

  void set_job_name(std::string job_name) {
    job_name_ = job_name;
  }
 private:
  std::string job_name_;
  int64_t parallel_size_;
  
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_
