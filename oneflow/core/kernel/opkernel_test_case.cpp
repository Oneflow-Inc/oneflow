#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/kernel.h"
#include <random>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {

namespace {

std::string ExpectedBlobName(const std::string& name) {
  return name + "_$expected$";
}

}  // namespace

#if defined(WITH_CUDA)

template<>
Blob* OpKernelTestCase<DeviceType::kCPU>::CreateBlob(const BlobDesc* blob_desc,
                                                     Regst* regst) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMallocHost(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(regst, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kCPU);
}

#endif

template<>
void OpKernelTestCase<DeviceType::kCPU>::BuildKernelCtx(KernelCtx* ctx) {
  ctx->device_ctx = new CpuDeviceCtx(-1);
}

template<>
void OpKernelTestCase<DeviceType::kCPU>::SyncStream(KernelCtx* ctx) {}

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestCase<device_type>::CreateBlobWithSpecifiedVal(
    const BlobDesc* blob_desc, std::vector<T> val, Regst* regst) {
  return CreateBlobWithSpecifiedValPtr(blob_desc, &(val[0]), regst);
}

template<>
template<typename T>
void OpKernelTestCase<DeviceType::kCPU>::CheckInitializeResult(
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
Blob* OpKernelTestCase<DeviceType::kCPU>::CreateBlobWithSpecifiedValPtr(
    const BlobDesc* blob_desc, T* val, Regst* regst) {
  Blob* ret = CreateBlob(blob_desc, regst);
  CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataContentField(),
                       cudaMemcpyHostToHost));
  return ret;
}

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestCase<device_type>::CreateBlobWithRandomVal(
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

template<DeviceType device_type>
void OpKernelTestCase<device_type>::EnrollBlobRegst(
    const std::string& blob_name, Regst* regst) {
  CHECK(bn_in_op2regst_.emplace(blob_name, regst).second);
}

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestCase<device_type>::InitBlob(const std::string& name,
                                              const BlobDesc* blob_desc,
                                              const std::vector<T>& val) {
  Blob* blob =
      CreateBlobWithSpecifiedVal<T>(blob_desc, val, bn_in_op2regst_[name]);
  CHECK(bn_in_op2blob_.emplace(name, blob).second);
  return blob;
}

template<DeviceType device_type>
template<typename T>
void OpKernelTestCase<device_type>::ForwardCheckBlob(const std::string& name,

                                                     const BlobDesc* blob_desc,
                                                     const std::vector<T>& val,
                                                     bool need_random_init) {
  forward_asserted_blob_names_.push_back(name);
  if (need_random_init) {
    Blob* blob = CreateBlobWithRandomVal<T>(blob_desc, bn_in_op2regst_[name]);
    CHECK(bn_in_op2blob_.emplace(name, blob).second);
  }
  Blob* blob =
      CreateBlobWithSpecifiedVal<T>(blob_desc, val, bn_in_op2regst_[name]);
  CHECK(bn_in_op2blob_.emplace(ExpectedBlobName(name), blob).second);
}

template<DeviceType device_type>
template<typename T>
void OpKernelTestCase<device_type>::ForwardCheckBlob(
    const std::string& name,

    const BlobDesc* blob_desc, const std::vector<T>& val) {
  ForwardCheckBlob(name, blob_desc, val, true);
}

template<DeviceType device_type>
template<typename T>
void OpKernelTestCase<device_type>::BackwardCheckBlob(const std::string& name,

                                                      const BlobDesc* blob_desc,
                                                      const std::vector<T>& val,
                                                      bool need_random_init) {
  backward_asserted_blob_names_.push_back(name);
  if (need_random_init) {
    Blob* blob = CreateBlobWithRandomVal<T>(blob_desc, bn_in_op2regst_[name]);
    CHECK(bn_in_op2blob_.emplace(name, blob).second);
  }
  Blob* blob =
      CreateBlobWithSpecifiedVal<T>(blob_desc, val, bn_in_op2regst_[name]);
  CHECK(bn_in_op2blob_.emplace(ExpectedBlobName(name), blob).second);
}

template<DeviceType device_type>
template<typename T>
void OpKernelTestCase<device_type>::BackwardCheckBlob(
    const std::string& name, const BlobDesc* blob_desc,
    const std::vector<T>& val) {
  BackwardCheckBlob(name, blob_desc, val, true);
}

template<DeviceType device_type>
Blob* OpKernelTestCase<device_type>::SwitchCreateBlobWithRandomVal(
    const BlobDesc* blob_desc, Regst* regst) {
  switch (blob_desc->data_type()) {
#define CREATE_BLOB_WITH_RANDOM_VAL_ENTRY(type_cpp, type_proto) \
  case type_proto: return CreateBlobWithRandomVal<type_cpp>(blob_desc, regst);
    OF_PP_FOR_EACH_TUPLE(CREATE_BLOB_WITH_RANDOM_VAL_ENTRY, ALL_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
  return nullptr;
}

template<DeviceType device_type>
std::function<Blob*(const std::string&)>
OpKernelTestCase<device_type>::MakeGetterBnInOp2Blob() {
  return [this](const std::string& bn_in_op) {
    if (bn_in_op2blob_[bn_in_op] == nullptr) {
      bn_in_op2blob_[bn_in_op] = SwitchCreateBlobWithRandomVal(
          &bn_in_op2blob_desc_.at(bn_in_op), bn_in_op2regst_[bn_in_op]);
    }
    return bn_in_op2blob_.at(bn_in_op);
  };
}

template<DeviceType device_type>
OpKernelTestCase<device_type>::OpKernelTestCase() {
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  parallel_ctx_.set_policy(ParallelPolicy::kModelParallel);
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::InitJobConf(
    const std::function<void(JobConf*)>& Init) {
  Init(&job_conf_);
  UpdateGlobalJobDesc();
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::UpdateGlobalJobDesc() {
  JobDescProto job_desc_proto;
  *job_desc_proto.mutable_job_conf() = job_conf_;
  if (Global<JobDesc>::Get()) { Global<JobDesc>::Delete(); }
  Global<JobDesc>::New(job_desc_proto);
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::InitBeforeRun() {
  for (const auto& pair : bn_in_op2blob_) {
    bn_in_op2blob_desc_[pair.first] = pair.second->blob_desc();
  }
  BuildKernelCtx(&kernel_ctx_);
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::set_is_train(bool is_train) {
  if (is_train) {
    job_conf_.mutable_train_conf();
  } else {
    job_conf_.mutable_predict_conf();
  }
  UpdateGlobalJobDesc();
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::AssertAfterRun() const {
  const std::list<std::string>* asserted_blob_names = nullptr;
  if (Global<JobDesc>::Get()->IsPredict() || is_forward_) {
    asserted_blob_names = &forward_asserted_blob_names_;
  } else {
    asserted_blob_names = &backward_asserted_blob_names_;
  }
  for (const auto& blob_name : *asserted_blob_names) {
    SwitchBlobCmp(blob_name, bn_in_op2blob_.at(blob_name),
                  bn_in_op2blob_.at(ExpectedBlobName(blob_name)));
  }
}

template<>
template<typename T>
void OpKernelTestCase<DeviceType::kCPU>::BlobCmp(const std::string& blob_name,
                                                 const Blob* lhs,
                                                 const Blob* rhs) {
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

template<DeviceType device_type>
void OpKernelTestCase<device_type>::SwitchBlobCmp(const std::string& blob_name,
                                                  const Blob* lhs,
                                                  const Blob* rhs) {
  switch (lhs->data_type()) {
#define BLOB_CMP_ENTRY(type_cpp, type_proto) \
  case type_proto: return BlobCmp<type_cpp>(blob_name, lhs, rhs);
    OF_PP_FOR_EACH_TUPLE(BLOB_CMP_ENTRY, ALL_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::SwitchCheckInitializeResult(
    const Blob* blob, const InitializerConf& initializer_conf) {
  switch (blob->data_type()) {
#define CHECK_INITIALIZED_RESULT_ENTRY(type_cpp, type_proto) \
  case type_proto:                                           \
    return CheckInitializeResult<type_cpp>(blob, initializer_conf);
    OF_PP_FOR_EACH_TUPLE(CHECK_INITIALIZED_RESULT_ENTRY, ALL_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::Run() {
  InitBeforeRun();
  auto op = ConstructOp(op_conf_);
  auto BnInOp2BlobDesc = MakeGetterBnInOp2BlobDesc();
  op->InferBlobDescs(BnInOp2BlobDesc, &parallel_ctx_, device_type);
  auto BnInOp2Blob = MakeGetterBnInOp2Blob();
  auto Launch = [&](bool is_forward) {
    KernelConf kernel_conf;
    op->GenKernelConf(BnInOp2BlobDesc, is_forward, device_type, &parallel_ctx_,
                      &kernel_conf, nullptr);
    auto kernel = ConstructKernel(&parallel_ctx_, kernel_conf);
    kernel->Launch(kernel_ctx_, BnInOp2Blob);
    SyncStream(&kernel_ctx_);
  };
  Launch(this);
  if (Global<JobDesc>::Get()->IsTrain() && !is_forward_) {
    initiation_before_backward_();
    Launch(false);
  }
  AssertAfterRun();
}

template<DeviceType device_type>
std::function<BlobDesc*(const std::string&)>
OpKernelTestCase<device_type>::MakeGetterBnInOp2BlobDesc() {
  return [this](const std::string& bn_in_op) {
    return &bn_in_op2blob_desc_[bn_in_op];
  };
}

#define INSTANTIATE_OPKERNEL_TEST_CASE(device_type) \
  template class OpKernelTestCase<device_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OPKERNEL_TEST_CASE, DEVICE_TYPE_SEQ);

#define INSTANTIATE_OPKERNEL_TEST_CASE_METHODS(device_type, data_type_pair)  \
  template Blob*                                                             \
  OpKernelTestCase<device_type>::InitBlob<OF_PP_PAIR_FIRST(data_type_pair)>( \
      const std::string&, const BlobDesc* blob_desc,                         \
      const std::vector<OF_PP_PAIR_FIRST(data_type_pair)>& val);             \
  template void OpKernelTestCase<device_type>::ForwardCheckBlob<             \
      OF_PP_PAIR_FIRST(data_type_pair)>(                                     \
      const std::string&, const BlobDesc* blob_desc,                         \
      const std::vector<OF_PP_PAIR_FIRST(data_type_pair)>& val);             \
  template void OpKernelTestCase<device_type>::BackwardCheckBlob<            \
      OF_PP_PAIR_FIRST(data_type_pair)>(                                     \
      const std::string&, const BlobDesc* blob_desc,                         \
      const std::vector<OF_PP_PAIR_FIRST(data_type_pair)>& val);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_OPKERNEL_TEST_CASE_METHODS,
                                 DEVICE_TYPE_SEQ, ALL_DATA_TYPE_SEQ);

}  // namespace test

}  // namespace oneflow
