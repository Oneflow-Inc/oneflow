#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/kernel.h"
#include <random>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {

namespace {

#define MAKE_OPKT_SWITCH_ENTRY_2(func_name, device_type, T) \
  OpKernelTestUtil<device_type>::template func_name<T>

#define DEFINE_OPKT_STATIC_SWITCH_FUNC(return_type, func_name)                \
  DEFINE_STATIC_SWITCH_FUNC(return_type, func_name, MAKE_OPKT_SWITCH_ENTRY_2, \
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),       \
                            MAKE_DATA_TYPE_CTRV_SEQ(ALL_DATA_TYPE_SEQ))

DEFINE_OPKT_STATIC_SWITCH_FUNC(void, BlobCmp);
DEFINE_OPKT_STATIC_SWITCH_FUNC(Blob*, CreateBlobWithRandomVal);
DEFINE_OPKT_STATIC_SWITCH_FUNC(void, CheckInitializeResult);

#define MAKE_OPKT_SWITCH_ENTRY_1(func_name, device_type) \
  OpKernelTestUtil<device_type>::func_name
DEFINE_STATIC_SWITCH_FUNC(void, BuildKernelCtx, MAKE_OPKT_SWITCH_ENTRY_1,
                          MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ))
DEFINE_STATIC_SWITCH_FUNC(void, SyncStream, MAKE_OPKT_SWITCH_ENTRY_1,
                          MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ))

bool NeedInferBlobDescs(Operator* op) {
  static const HashSet<int> no_need_infer_op{OperatorConf::kCopyHdConf};
  return no_need_infer_op.find(op->op_conf().op_type_case())
         == no_need_infer_op.end();
}

}  // namespace

#define DEFINE_OPKT_UTIL_STATIC_SWITCH_FUNC(return_type, func_name) \
  DEFINE_STATIC_SWITCH_FUNC(return_type, func_name,                 \
                            MAKE_OPKT_UTIL_SWITCH_ENTRY,            \
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ))

template<typename T>
struct OpKTSwitchHelper final {
#define MAKE_OPKT_UTIL_SWITCH_ENTRY(func_name, device_type) \
  OpKernelTestUtil<device_type>::template func_name<T>
  DEFINE_OPKT_UTIL_STATIC_SWITCH_FUNC(Blob*, CreateBlobWithRandomVal);
  DEFINE_OPKT_UTIL_STATIC_SWITCH_FUNC(Blob*, CreateBlobWithSpecifiedVal);
};

#if defined(WITH_CUDA)

template<>
Blob* OpKernelTestUtil<DeviceType::kCPU>::CreateBlob(const BlobDesc* blob_desc,
                                                     Regst* regst) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMallocHost(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(regst, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kCPU);
}

#endif

template<>
void OpKernelTestUtil<DeviceType::kCPU>::BuildKernelCtx(KernelCtx* ctx) {
  ctx->device_ctx = new CpuDeviceCtx(-1);
}

template<>
void OpKernelTestUtil<DeviceType::kCPU>::SyncStream(KernelCtx* ctx) {}

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestUtil<device_type>::CreateBlobWithSpecifiedVal(
    const BlobDesc* blob_desc, std::vector<T> val, Regst* regst) {
  return CreateBlobWithSpecifiedValPtr(blob_desc, &(val[0]), regst);
}

template<>
template<typename T>
void OpKernelTestUtil<DeviceType::kCPU>::CheckInitializeResult(
    const Blob* blob, const InitializerConf& initializer_conf) {
  if (initializer_conf.has_constant_conf()) {
    for (int64_t i = 0; i < blob->shape().elem_cnt(); ++i) {
      ASSERT_FLOAT_EQ(blob->dptr<T>()[i],
                      initializer_conf.constant_conf().value());
    }
  } else if (initializer_conf.has_random_uniform_conf()) {
    TODO();
  } else if (initializer_conf.has_random_normal_conf()) {
    TODO();
  } else {
    UNIMPLEMENTED();
  }
}

template<>
template<typename T>
Blob* OpKernelTestUtil<DeviceType::kCPU>::CreateBlobWithSpecifiedValPtr(
    const BlobDesc* blob_desc, T* val, Regst* regst) {
  Blob* ret = CreateBlob(blob_desc, regst);
  CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataContentField(),
                       cudaMemcpyHostToHost));
  return ret;
}

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestUtil<device_type>::CreateBlobWithRandomVal(
    const BlobDesc* blob_desc, Regst* regst) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 10);
  std::vector<T> val_vec(blob_desc->shape().elem_cnt());
  for (int64_t i = 0; i < blob_desc->shape().elem_cnt(); ++i) {
    val_vec[i] = static_cast<T>(dis(gen));
  }
  return CreateBlobWithSpecifiedVal<T>(blob_desc, val_vec, regst);
}

template<>
template<typename T>
void OpKernelTestUtil<DeviceType::kCPU>::BlobCmp(const std::string& blob_name,
                                                 const Blob* lhs,
                                                 DeviceType lhs_device_type,
                                                 const Blob* rhs,
                                                 DeviceType rhs_device_type) {
  CHECK_EQ(lhs_device_type, DeviceType::kCPU);
  CHECK_EQ(rhs_device_type, DeviceType::kCPU);
  ASSERT_EQ(lhs->blob_desc(), rhs->blob_desc()) << blob_name;
  CHECK_EQ(lhs->data_type(), GetDataType<T>::value);
  if (IsFloatingDataType(lhs->data_type())) {
    for (int64_t i = 0; i < lhs->shape().elem_cnt(); ++i) {
      ASSERT_NEAR(lhs->dptr<T>()[i], rhs->dptr<T>()[i], 1e-5) << blob_name;
    }
  } else {
    ASSERT_EQ(
        memcmp(lhs->dptr(), rhs->dptr(), lhs->ByteSizeOfDataContentField()), 0)
        << blob_name;
  }
}

#define INSTANTIATE_CPU_OPKERNEL_TEST_UTIL_METHODS(T, data_type_proto)        \
  template void OpKernelTestUtil<DeviceType::kCPU>::CheckInitializeResult<T>( \
      const Blob* blob, const InitializerConf& initializer_conf);

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_OPKERNEL_TEST_UTIL_METHODS,
                     ALL_DATA_TYPE_SEQ);

void OpKernelTestCase::EnrollBlobRegst(const std::string& blob_name,
                                       Regst* regst) {
  CHECK(bn_in_op2regst_.emplace(blob_name, regst).second);
}

template<typename T>
Blob* OpKernelTestCase::InitBlob(const std::string& name,
                                 const BlobDesc* blob_desc,
                                 const std::vector<T>& val) {
  DeviceType dev_type = GetBlobDeviceType(name);
  Blob* blob = OpKTSwitchHelper<T>::SwitchCreateBlobWithSpecifiedVal(
      SwitchCase(dev_type), blob_desc, val, bn_in_op2regst_[name]);
  CHECK(bn_in_op2blob_.emplace(name, blob).second);
  return blob;
}

template<typename T>
Blob* OpKernelTestCase::RandomInitBlob(const std::string& name,
                                       const BlobDesc* blob_desc) {
  DeviceType dev_type = GetBlobDeviceType(name);
  Blob* blob = OpKTSwitchHelper<T>::SwitchCreateBlobWithRandomVal(
      SwitchCase(dev_type), blob_desc, bn_in_op2regst_[name]);
  CHECK(bn_in_op2blob_.emplace(name, blob).second);
  return blob;
}

template<typename T>
void OpKernelTestCase::CheckBlob(const std::string& name,
                                 const BlobDesc* blob_desc,
                                 const std::vector<T>& val,
                                 bool need_random_init) {
  DeviceType dev_type = GetBlobDeviceType(name);
  if (need_random_init) {
    Blob* blob = OpKTSwitchHelper<T>::SwitchCreateBlobWithRandomVal(
        SwitchCase(dev_type), blob_desc, bn_in_op2regst_[name]);
    CHECK(bn_in_op2blob_.emplace(name, blob).second);
  }
  Blob* expected_blob = OpKTSwitchHelper<T>::SwitchCreateBlobWithSpecifiedVal(
      SwitchCase(dev_type), blob_desc, val, bn_in_op2regst_[name]);
  CHECK(bn_in_op2blob_.emplace(ExpectedBlobName(name), expected_blob).second);
}

#define __LOC__() (__FILE__ ":" OF_PP_STRINGIZE(__LINE__))

template<typename T>
void OpKernelTestCase::CheckBlob(const std::string& name,
                                 const BlobDesc* blob_desc,
                                 const std::string& expected_existed_blob_name,
                                 bool need_random_init) {
  DeviceType dev_type = GetBlobDeviceType(name);
  if (need_random_init) {
    Blob* blob = OpKTSwitchHelper<T>::SwitchCreateBlobWithRandomVal(
        SwitchCase(dev_type), blob_desc, bn_in_op2regst_[name]);
    CHECK(bn_in_op2blob_.emplace(name, blob).second);
  }
  CHECK(bn_in_op2blob_.find(expected_existed_blob_name) != bn_in_op2blob_.end())
      << __LOC__() << ": " << expected_existed_blob_name;
  Blob* expected_blob = bn_in_op2blob_.at(expected_existed_blob_name);
  SetBlobSpecializedDeviceType(ExpectedBlobName(name),
                               GetBlobDeviceType(expected_existed_blob_name));
  CHECK(bn_in_op2blob_.emplace(ExpectedBlobName(name), expected_blob).second);
}

template<typename T>
void OpKernelTestCase::ForwardCheckBlob(const std::string& name,
                                        const BlobDesc* blob_desc,
                                        const std::vector<T>& val,
                                        bool need_random_init) {
  forward_asserted_blob_names_.push_back(name);
  CheckBlob(name, blob_desc, val, need_random_init);
}

template<typename T>
void OpKernelTestCase::ForwardCheckBlob(
    const std::string& name, const BlobDesc* blob_desc,
    const std::string& expected_exist_blob_name, bool need_random_init) {
  forward_asserted_blob_names_.push_back(name);
  CheckBlob<T>(name, blob_desc, expected_exist_blob_name, need_random_init);
}

template<typename T>
void OpKernelTestCase::ForwardCheckBlob(const std::string& name,
                                        const BlobDesc* blob_desc,
                                        const std::vector<T>& val) {
  ForwardCheckBlob(name, blob_desc, val, true);
}

template<typename T>
void OpKernelTestCase::BackwardCheckBlob(const std::string& name,
                                         const BlobDesc* blob_desc,
                                         const std::vector<T>& val,
                                         bool need_random_init) {
  backward_asserted_blob_names_.push_back(name);
  CheckBlob(name, blob_desc, val, need_random_init);
}

template<typename T>
void OpKernelTestCase::BackwardCheckBlob(
    const std::string& name, const BlobDesc* blob_desc,
    const std::string& expected_exist_blob_name, bool need_random_init) {
  backward_asserted_blob_names_.push_back(name);
  CheckBlob<T>(name, blob_desc, expected_exist_blob_name, need_random_init);
}

template<typename T>
void OpKernelTestCase::BackwardCheckBlob(const std::string& name,
                                         const BlobDesc* blob_desc,
                                         const std::vector<T>& val) {
  BackwardCheckBlob(name, blob_desc, val, true);
}

std::function<Blob*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2Blob() {
  return [this](const std::string& bn_in_op) -> Blob* {
    if (bn_in_op2blob_[bn_in_op] == nullptr) {
      const auto& it = bn_in_op2blob_desc_.find(bn_in_op);
      if (it == bn_in_op2blob_desc_.end()) { return nullptr; }
      DeviceType dev_type = GetBlobDeviceType(bn_in_op);
      const BlobDesc* blob_desc = &it->second;
      bn_in_op2blob_[bn_in_op] = SwitchCreateBlobWithRandomVal(
          SwitchCase(dev_type, blob_desc->data_type()), blob_desc,
          bn_in_op2regst_[bn_in_op]);
    }

    CHECK(bn_in_op2blob_.find(bn_in_op) != bn_in_op2blob_.end())
        << __LOC__() << ": " << bn_in_op;
    return bn_in_op2blob_.at(bn_in_op);
  };
}

OpKernelTestCase::OpKernelTestCase() {
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  parallel_ctx_.set_policy(ParallelPolicy::kModelParallel);
  initiation_before_backward_ = []() {};
}

void OpKernelTestCase::InitJobConf(const std::function<void(JobConf*)>& Init) {
  Init(&job_conf_);
  UpdateGlobalJobDesc();
}

void OpKernelTestCase::UpdateGlobalJobDesc() {
  JobDescProto job_desc_proto;
  *job_desc_proto.mutable_job_conf() = job_conf_;
  if (Global<JobDesc>::Get()) { Global<JobDesc>::Delete(); }
  Global<JobDesc>::New(job_desc_proto);
}

void OpKernelTestCase::InitBeforeRun() {
  for (const auto& pair : bn_in_op2blob_) {
    bn_in_op2blob_desc_[pair.first] = pair.second->blob_desc();
  }
  SwitchBuildKernelCtx(SwitchCase(default_device_type()), &kernel_ctx_);
}

void OpKernelTestCase::set_is_train(bool is_train) {
  if (is_train) {
    job_conf_.mutable_train_conf();
  } else {
    job_conf_.mutable_predict_conf();
  }
  UpdateGlobalJobDesc();
}

void OpKernelTestCase::AssertAfterRun() const {
  const std::list<std::string>* asserted_blob_names = nullptr;
  if (Global<JobDesc>::Get()->IsPredict() || is_forward_) {
    asserted_blob_names = &forward_asserted_blob_names_;
  } else {
    asserted_blob_names = &backward_asserted_blob_names_;
  }
  for (const auto& blob_name : *asserted_blob_names) {
    DeviceType lhs_dev_type = GetBlobDeviceType(blob_name);
    DeviceType rhs_dev_type = GetBlobDeviceType(ExpectedBlobName(blob_name));
    DeviceType dev_type = ((lhs_dev_type == DeviceType::kGPU)
                                   || (rhs_dev_type == DeviceType::kGPU)
                               ? DeviceType::kGPU
                               : DeviceType::kCPU);

    CHECK(bn_in_op2blob_.find(blob_name) != bn_in_op2blob_.end())
        << __LOC__() << ": " << blob_name;
    CHECK(bn_in_op2blob_.find(ExpectedBlobName(blob_name))
          != bn_in_op2blob_.end())
        << __LOC__() << ": " << ExpectedBlobName(blob_name);
    DataType data_type = bn_in_op2blob_.at(blob_name)->data_type();

    SwitchBlobCmp(SwitchCase(dev_type, data_type), blob_name,
                  bn_in_op2blob_.at(blob_name), lhs_dev_type,
                  bn_in_op2blob_.at(ExpectedBlobName(blob_name)), rhs_dev_type);
  }
}

void OpKernelTestCase::Run() {
  InitBeforeRun();
  auto op = ConstructOp(op_conf_);
  auto BnInOp2BlobDesc = MakeGetterBnInOp2BlobDesc();
  OpContext* op_context = nullptr;
  if (NeedInferBlobDescs(op.get())) {
    op->InferBlobDescs(BnInOp2BlobDesc, &parallel_ctx_, default_device_type(),
                       [&](OpContext* op_ctx) { op_context = op_ctx; });
  }
  auto BnInOp2Blob = MakeGetterBnInOp2Blob();
  auto Launch = [&](bool is_forward) {
    KernelConf kernel_conf;
    op->GenKernelConf(BnInOp2BlobDesc, is_forward, default_device_type(),
                      &parallel_ctx_, &kernel_conf, op_context);
    auto kernel = ConstructKernel(&parallel_ctx_, kernel_conf);
    kernel->Launch(kernel_ctx_, BnInOp2Blob);
    SwitchSyncStream(SwitchCase(default_device_type()), &kernel_ctx_);
  };
  Launch(this);
  if (Global<JobDesc>::Get()->IsTrain() && !is_forward_) {
    initiation_before_backward_();
    Launch(false);
  }
  AssertAfterRun();
}

std::function<BlobDesc*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2BlobDesc() {
  return [this](const std::string& bn_in_op) {
    return &bn_in_op2blob_desc_[bn_in_op];
  };
}

#define INSTANTIATE_OPKERNEL_TEST_CASE_METHODS(T, type_proto)              \
  template Blob* OpKernelTestCase::InitBlob<T>(const std::string&,         \
                                               const BlobDesc* blob_desc,  \
                                               const std::vector<T>& val); \
  template Blob* OpKernelTestCase::RandomInitBlob<T>(                      \
      const std::string&, const BlobDesc* blob_desc);                      \
  template void OpKernelTestCase::ForwardCheckBlob<T>(                     \
      const std::string&, const BlobDesc* blob_desc,                       \
      const std::vector<T>& val);                                          \
  template void OpKernelTestCase::ForwardCheckBlob<T>(                     \
      const std::string&, const BlobDesc* blob_desc, const std::string&,   \
      bool);                                                               \
  template void OpKernelTestCase::BackwardCheckBlob<T>(                    \
      const std::string&, const BlobDesc* blob_desc,                       \
      const std::vector<T>& val);                                          \
  template void OpKernelTestCase::BackwardCheckBlob<T>(                    \
      const std::string&, const BlobDesc* blob_desc, const std::string&,   \
      bool);

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OPKERNEL_TEST_CASE_METHODS, ALL_DATA_TYPE_SEQ);

}  // namespace test

}  // namespace oneflow
