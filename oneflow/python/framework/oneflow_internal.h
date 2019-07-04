#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/common/buffer_manager.h"
#include <google/protobuf/text_format.h>

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

void LaunchJob(const std::string& job_name, const std::shared_ptr<oneflow::ForeignCallback>& cb) {
  using namespace oneflow;
  CHECK_NOTNULL(Global<Oneflow>::Get());
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignCallback>>>::Get();
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

int Ofblob_GetElemCnt(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->elem_cnt();
}

size_t OfBlob_NumAxes(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumAxes();
}
void OfBlob_CopyShapeTo(uint64_t of_blob_ptr, int64_t* ptr, int64_t num_axis) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(ptr, num_axis);
};

#define DEFINE_COPIER(T, type_proto)                                                \
  void OfBlob_CopyToBuffer_##T(uint64_t of_blob_ptr, T* ptr, int64_t len) {         \
    using namespace oneflow;                                                        \
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);                         \
    of_blob->AutoMemCopyTo<T>(ptr, len);                                            \
  }                                                                                 \
  void OfBlob_CopyFromBuffer_##T(uint64_t of_blob_ptr, const T* ptr, int64_t len) { \
    using namespace oneflow;                                                        \
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);                         \
    of_blob->AutoMemCopyFrom<T>(ptr, len);                                          \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_COPIER, POD_DATA_TYPE_SEQ);

#undef DEFINE_COPIER

std::string OfBlob_GetCopyToBufferFuncName(uint64_t of_blob_ptr) {
  using namespace oneflow;
  static const HashMap<int64_t, std::string> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, "OfBlob_CopyToBuffer_" #type_cpp},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return data_type2func_name.at(of_blob->data_type());
}

std::string OfBlob_GetCopyFromBufferFuncName(uint64_t of_blob_ptr) {
  using namespace oneflow;
  static const HashMap<int64_t, std::string> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, "OfBlob_CopyFromBuffer_" #type_cpp},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return data_type2func_name.at(of_blob->data_type());
}

template<typename T>
void CopyFromNdarry(T* array, int size, uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<T>(array, size);
}

template<typename T>
void CopyToNdarry(T* array, int size, uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<T>(array, size);
}
namespace {

struct GlobalOneflowChecker final {
  GlobalOneflowChecker() = default;
  ~GlobalOneflowChecker() {
    using namespace oneflow;
    if (Global<Oneflow>::Get() == nullptr) {
      LOG(INFO) << "oneflow exits successfully";
    } else {
      LOG(FATAL) << "Global<Oneflow> is not destroyed yet";
    }
  }
};

GlobalOneflowChecker checker;

}  // namespace
