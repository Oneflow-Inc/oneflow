#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/common/buffer_manager.h"

void NaiveSequentialRunSerializedJobSet(const oneflow::JobSet& job_set) {
  using namespace oneflow;
  CHECK_ISNULL(Global<Oneflow>::Get());
  Main(job_set);
}

void InitGlobalOneflowBySerializedJobSet(const oneflow::JobSet& job_set) {
  using namespace oneflow;
  CHECK_ISNULL(Global<Oneflow>::Get());
  Global<Oneflow>::New(job_set);
}

std::string GetSerializedInterUserJobInfo() {
  using namespace oneflow;
  CHECK_NOTNULL(Global<Oneflow>::Get());
  CHECK_NOTNULL(Global<InterUserJobInfo>::Get());
  std::string ret;
  google::protobuf::TextFormat::PrintToString(*Global<InterUserJobInfo>::Get(), &ret);
  return ret;
}

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  using namespace oneflow;
  CHECK_NOTNULL(Global<Oneflow>::Get());
  const auto& job_name = cb->job_name();
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  int64_t job_id = Global<JobName2JobId>::Get()->at(job_name);
  const JobDesc& job_desc = GlobalJobDesc(job_id);
  if (job_desc.is_pull_job()) { buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Send(cb); }
  if (job_desc.is_push_job()) { buffer_mgr->Get(GetForeignInputBufferName(job_name))->Send(cb); }
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(cb);
  Global<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Send(job_id);
}

void DestroyGlobalOneflow() {
  using namespace oneflow;
  CHECK_NOTNULL(Global<Oneflow>::Get());
  Global<Oneflow>::Delete();
}

int Ofblob_GetDataType(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->data_type();
}

size_t OfBlob_NumAxes(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumAxes();
}

void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, int64_t* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(array, size);
};

namespace {

struct GlobalOneflowChecker final {
  GlobalOneflowChecker() = default;
  ~GlobalOneflowChecker() {
    using namespace oneflow;
    if (Global<Oneflow>::Get() != nullptr) { LOG(FATAL) << "global oneflow is not destroyed yet"; }
  }
};

GlobalOneflowChecker checker;

}  // namespace
