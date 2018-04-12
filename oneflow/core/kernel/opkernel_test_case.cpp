#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

namespace test {

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

  static constexpr DataType GetDataTypeValue() { return GetDataType<T>::value; }
};

namespace {

#define DEFINE_OPKT_SWITCHER(return_type, func_name)                          \
  DEFINE_STATIC_SWITCH_FUNC(return_type, func_name, MAKE_OPKT_SWITCH_ENTRY_2, \
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),       \
                            MAKE_DATA_TYPE_CTRV_SEQ(ALL_DATA_TYPE_SEQ))
#define MAKE_OPKT_SWITCH_ENTRY_2(func_name, device_type, T) \
  OpKernelTestUtil<device_type>::template func_name<T>

DEFINE_OPKT_SWITCHER(void, BlobCmp);
DEFINE_OPKT_SWITCHER(Blob*, CreateBlobWithRandomVal);
DEFINE_OPKT_SWITCHER(void, CheckInitializeResult);

#define DEFINE_OPKT_DEV_SWITCHER(return_type, func_name)                      \
  DEFINE_STATIC_SWITCH_FUNC(return_type, func_name, MAKE_OPKT_SWITCH_ENTRY_1, \
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ))
#define MAKE_OPKT_SWITCH_ENTRY_1(func_name, device_type) \
  OpKernelTestUtil<device_type>::func_name

DEFINE_OPKT_DEV_SWITCHER(void, BuildKernelCtx);
DEFINE_OPKT_DEV_SWITCHER(void, SyncStream);
DEFINE_OPKT_DEV_SWITCHER(Blob*, CreateBlob);

bool NeedInferBlobDescs(Operator* op) {
  static const HashSet<int> no_need_infer_op{OperatorConf::kCopyHdConf};
  return no_need_infer_op.find(op->op_conf().op_type_case())
         == no_need_infer_op.end();
}

void BlobCmp(const std::string& blob_name, const Blob* lhs,
             DeviceType lhs_device_type, const Blob* rhs,
             DeviceType rhs_device_type) {
  CHECK_EQ(lhs->data_type(), rhs->data_type());
  DeviceType dev_type = DeviceType::kCPU;
  if (lhs_device_type == DeviceType::kGPU) { dev_type = DeviceType::kGPU; }
  if (rhs_device_type == DeviceType::kGPU) { dev_type = DeviceType::kGPU; }
  SwitchBlobCmp(SwitchCase(dev_type, lhs->data_type()), blob_name, lhs,
                lhs_device_type, rhs, rhs_device_type);
}

#define MAKE_OPK_HELPER_SWITCH_ENTRY(func_name, type_cpp) \
  OpKTSwitchHelper<type_cpp>::func_name
DEFINE_STATIC_SWITCH_FUNC(DataType, GetDataTypeValue,
                          MAKE_OPK_HELPER_SWITCH_ENTRY,
                          MAKE_STRINGIZED_DATA_TYPE_CTRV_SEQ(ALL_DATA_TYPE_SEQ))

}  // namespace

DataType DataType4CppTypeString(const std::string& cpp_type_str) {
  return SwitchGetDataTypeValue(SwitchCase(cpp_type_str));
}

#if defined(WITH_CUDA)

template<>
Blob* OpKernelTestUtil<DeviceType::kCPU>::CreateBlob(const BlobDesc* blob_desc,
                                                     Regst* regst) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMallocHost(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(regst, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kCPU);
}

template<DeviceType src_device_type, DeviceType dst_device_type>
struct GetCudaMemCpyKind;

template<>
struct GetCudaMemCpyKind<DeviceType::kCPU, DeviceType::kCPU> final {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyHostToHost;
};

template<>
struct GetCudaMemCpyKind<DeviceType::kCPU, DeviceType::kGPU> final {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyHostToDevice;
};

template<>
struct GetCudaMemCpyKind<DeviceType::kGPU, DeviceType::kGPU> final {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};

template<>
struct GetCudaMemCpyKind<DeviceType::kGPU, DeviceType::kCPU> final {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyDeviceToHost;
};

template<DeviceType dst_device_type, DeviceType src_device_type>
void BlobCopy(Blob* dst, const Blob* src) {
  CHECK_EQ(dst->ByteSizeOfDataContentField(),
           src->ByteSizeOfDataContentField());
  CudaCheck(cudaMemcpy(
      dst->mut_dptr(), src->dptr(), dst->ByteSizeOfDataContentField(),
      GetCudaMemCpyKind<src_device_type, dst_device_type>::val));
}

#define MAKE_TWO_DEVICE_SWITCH_ENTRY(func_name, dev_type0, dev_type1) \
  func_name<dev_type0, dev_type1>
DEFINE_STATIC_SWITCH_FUNC(void, BlobCopy, MAKE_TWO_DEVICE_SWITCH_ENTRY,
                          MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                          MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ));

#endif

template<>
void OpKernelTestUtil<DeviceType::kCPU>::BuildKernelCtx(KernelCtx* ctx) {
  ctx->device_ctx = new CpuDeviceCtx(-1);
}

template<>
void OpKernelTestUtil<DeviceType::kGPU>::BuildKernelCtx(KernelCtx* ctx) {
  if (!Global<CudaStreamHandle>::Get()) { Global<CudaStreamHandle>::New(); }
  CudaStreamHandle* cuda_handle = Global<CudaStreamHandle>::Get();
  ctx->device_ctx = new CudaDeviceCtx(
      -1, cuda_handle->cuda_stream(), cuda_handle->cublas_pmh_handle(),
      cuda_handle->cublas_pmd_handle(), cuda_handle->cudnn_handle(),
      cuda_handle->eigen_gpu_device());
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
void OpKernelTestCase::CheckBlobWithAnother(
    const std::string& name, const BlobDesc* blob_desc,
    const std::string& expected_existed_blob_name, bool need_random_init) {
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
void OpKernelTestCase::ForwardCheckBlobWithAnother(
    const std::string& name, const BlobDesc* blob_desc,
    const std::string& expected_exist_blob_name, bool need_random_init) {
  forward_asserted_blob_names_.push_back(name);
  CheckBlobWithAnother<T>(name, blob_desc, expected_exist_blob_name,
                          need_random_init);
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
void OpKernelTestCase::BackwardCheckBlobWithAnother(
    const std::string& name, const BlobDesc* blob_desc,
    const std::string& expected_exist_blob_name, bool need_random_init) {
  backward_asserted_blob_names_.push_back(name);
  CheckBlobWithAnother<T>(name, blob_desc, expected_exist_blob_name,
                          need_random_init);
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

OpKernelTestCase::OpKernelTestCase()
    : is_forward_(false),
      default_device_type_(DeviceType::kInvalidDevice),
      initiation_before_backward_([]() {}) {
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  parallel_ctx_.set_policy(ParallelPolicy::kModelParallel);
}

void OpKernelTestCase::InitJobConf(const std::function<void(JobConf*)>& Init) {
  Init(&job_conf_);
  UpdateGlobalJobDesc();
}

void OpKernelTestCase::set_default_data_type(DataType default_data_type) {
  InitJobConf([=](JobConf* job_conf) {
    job_conf->set_default_data_type(default_data_type);
  });
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
    BlobCmp(blob_name, bn_in_op2blob_.at(blob_name), lhs_dev_type,
            bn_in_op2blob_.at(ExpectedBlobName(blob_name)), rhs_dev_type);
  }
}

Regst* OpKernelTestCase::GetBlobRegst(const std::string& bn_in_op) {
  const auto& it = bn_in_op2regst_.find(bn_in_op);
  if (it == bn_in_op2regst_.end()) { return nullptr; }
  return it->second;
}

std::function<BlobDesc*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2BlobDesc() {
  return [this](const std::string& bn) { return MutBlobDesc4BnInOp(bn); };
}

void OpKernelTestCase::InferBlobDesc(std::shared_ptr<Operator>* op,
                                     OpContext** op_context) {
  *op = ConstructOp(op_conf_);
  if (NeedInferBlobDescs(op->get())) {
    (*op)->InferBlobDescs(MakeGetterBnInOp2BlobDesc(), &parallel_ctx_,
                          default_device_type(),
                          [&](OpContext* op_ctx) { *op_context = op_ctx; });
  }
}

void OpKernelTestCase::RunKernel(Operator* op, OpContext* op_context) {
  auto Launch = [&](bool is_forward) {
    KernelConf kernel_conf;
    op->GenKernelConf(MakeGetterBnInOp2BlobDesc(), is_forward,
                      default_device_type(), &parallel_ctx_, &kernel_conf,
                      op_context);
    auto kernel = ConstructKernel(&parallel_ctx_, kernel_conf);
    kernel->Launch(kernel_ctx_, MakeGetterBnInOp2Blob());
    SwitchSyncStream(SwitchCase(default_device_type()), &kernel_ctx_);
  };
  Launch(true);
  if (Global<JobDesc>::Get()->IsTrain() && !is_forward_) {
    initiation_before_backward_();
    Launch(false);
  }
}

void OpKernelTestCase::Run() {
  InitBeforeRun();
  std::shared_ptr<Operator> op;
  OpContext* op_context = nullptr;
  InferBlobDesc(&op, &op_context);
  RunKernel(op.get(), op_context);
  AssertAfterRun();
}

std::list<std::string> DiffKernelImplTestCase::AllInputBlobNames() const {
  std::list<std::string> all_input_blob_names(input_blob_names_);
  for (const auto& bn : output_diff_blob_names_) {
    all_input_blob_names.push_back(bn);
  }
  return all_input_blob_names;
}

std::list<std::string> DiffKernelImplTestCase::AllOutputBlobNamesWithValidBlob()
    const {
  std::list<std::string> all_output_blob_names(output_blob_names_);
  if (Global<JobDesc>::Get()->IsTrain() && !is_forward()) {
    for (const auto& bn : input_diff_blob_names_) {
      all_output_blob_names.push_back(bn);
    }
  }
  for (const auto& bn_in_op : all_output_blob_names) {
    CHECK(bn_in_op2blob().find(bn_in_op) != bn_in_op2blob().end());
  }
  return all_output_blob_names;
}

DiffKernelImplTestCase::DiffKernelImplTestCase(bool is_train, bool is_forward,
                                               DataType default_data_type) {
  set_is_train(is_train);
  set_is_forward(is_forward);
  set_default_data_type(default_data_type);
  initiate_kernel_ctx_ = [](const std::function<Blob*(const std::string&)>&) {};
}

void DiffKernelImplTestCase::SetBlobNames(
    const std::list<std::string>& input_bn_in_op,
    const std::list<std::string>& output_bn_in_op,
    const std::list<std::string>& output_diff_bn_in_op,
    const std::list<std::string>& input_diff_bn_in_op) {
  input_blob_names_ = input_bn_in_op;
  output_blob_names_ = output_bn_in_op;
  output_diff_blob_names_ = output_diff_bn_in_op;
  input_diff_blob_names_ = input_diff_bn_in_op;
}

void DiffKernelImplTestCase::SetInputBlobDesc(const std::string& bn_in_op,
                                              const Shape& shape,
                                              DataType data_type) {
  BlobDesc blob_desc(shape, data_type, false, false, 1);
  *MutBlobDesc4BnInOp(bn_in_op) = blob_desc;
}

void DiffKernelImplTestCase::RandomInitInputOrigin() {
  for (const auto& bn_in_op : AllInputBlobNames()) {
    const BlobDesc* blob_desc = BlobDesc4BnInOp(bn_in_op);
    CHECK(blob_desc);
    Blob* blob = SwitchCreateBlobWithRandomVal(
        SwitchCase(DeviceType::kCPU, blob_desc->data_type()), blob_desc,
        GetBlobRegst(bn_in_op));
    const std::string& bn = GetOriginInputBlobName(bn_in_op);
    CHECK(mut_bn_in_op2blob()->emplace(bn, blob).second);
    SetBlobSpecializedDeviceType(bn, DeviceType::kCPU);
  }
}

void DiffKernelImplTestCase::InitInputBlobs() {
  for (const auto& bn_in_op : AllInputBlobNames()) {
    const std::string& origin_input_bn = GetOriginInputBlobName(bn_in_op);
    Blob* origin_input_blob = bn_in_op2blob().at(origin_input_bn);
    const BlobDesc* blob_desc = BlobDesc4BnInOp(bn_in_op);
    Blob* blob = SwitchCreateBlob(SwitchCase(default_device_type()), blob_desc,
                                  GetBlobRegst(bn_in_op));
    SwitchBlobCopy(SwitchCase(default_device_type(), DeviceType::kCPU), blob,
                   origin_input_blob);
    mut_bn_in_op2blob()->emplace(bn_in_op, blob);
  }
}

void DiffKernelImplTestCase::DumpBlobs(const std::string& prefix) {
  for (const auto& bn_in_op : AllOutputBlobNamesWithValidBlob()) {
    Blob* blob = bn_in_op2blob().at(bn_in_op);
    mut_bn_in_op2blob()->emplace(prefix + bn_in_op, blob);
    SetBlobSpecializedDeviceType(prefix + bn_in_op, default_device_type());
  }
}

void DiffKernelImplTestCase::CheckMultiRunResults(
    const std::string& base_prefix,
    const std::list<std::string>& other_prefixes) const {
  for (const auto& checkee_prefix : other_prefixes) {
    for (const auto& bn_in_op : AllOutputBlobNamesWithValidBlob()) {
      const std::string& base_bn = base_prefix + bn_in_op;
      Blob* base_blob = bn_in_op2blob().at(base_bn);
      DeviceType base_blob_device_type = GetBlobDeviceType(base_bn);
      const std::string& checkee_bn = checkee_prefix + bn_in_op;
      Blob* checkee_blob = bn_in_op2blob().at(checkee_bn);
      DeviceType checkee_blob_device_type = GetBlobDeviceType(checkee_bn);
      BlobCmp(checkee_bn, checkee_blob, checkee_blob_device_type, base_blob,
              base_blob_device_type);
    }
  }
}

void DiffKernelImplTestCase::CopyBlobDesc4DiffBlob() {
  auto CopyBlobDesc = [&](const std::list<std::string>& blob_names) {
    for (const auto& bn_in_op : blob_names) {
      const BlobDesc* blob_desc = BlobDesc4BnInOp(bn_in_op);
      if (blob_desc) { *MutBlobDesc4BnInOp(GenDiffBn(bn_in_op)) = *blob_desc; }
    }
  };
  CopyBlobDesc(input_blob_names_);
  CopyBlobDesc(output_blob_names_);
}

void DiffKernelImplTestCase::MultiRunThenCheck() {
  std::shared_ptr<Operator> op;
  OpContext* op_context = nullptr;
  InferBlobDesc(&op, &op_context);
  CopyBlobDesc4DiffBlob();
  RandomInitInputOrigin();
  auto Run = [&](const std::string& dump_prefix) {
    std::shared_ptr<Operator> op;
    InferBlobDesc(&op, &op_context);
    InitInputBlobs();
    SwitchBuildKernelCtx(SwitchCase(default_device_type()), mut_kernel_ctx());
    initiate_kernel_ctx_(MakeGetterBnInOp2Blob());
    RunKernel(op.get(), op_context);
    DumpBlobs(dump_prefix);
  };
  set_default_device_type(DeviceType::kCPU);
  Run("cpu_");
  set_default_device_type(DeviceType::kGPU);
  Run("gpu_");
  mut_op_conf()->set_use_cudnn_on_gpu(true);
  Run("cudnn_");
  CheckMultiRunResults("cpu_", {"gpu_", "cudnn_"});
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
      const std::string&, const BlobDesc* blob_desc,                       \
      const std::vector<T>& val, bool need_random_init);                   \
  template void OpKernelTestCase::ForwardCheckBlobWithAnother<T>(          \
      const std::string&, const BlobDesc* blob_desc, const std::string&,   \
      bool);                                                               \
  template void OpKernelTestCase::BackwardCheckBlob<T>(                    \
      const std::string&, const BlobDesc* blob_desc,                       \
      const std::vector<T>& val);                                          \
  template void OpKernelTestCase::BackwardCheckBlob<T>(                    \
      const std::string&, const BlobDesc* blob_desc,                       \
      const std::vector<T>& val, bool need_random_init);                   \
  template void OpKernelTestCase::BackwardCheckBlobWithAnother<T>(         \
      const std::string&, const BlobDesc* blob_desc, const std::string&,   \
      bool);

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OPKERNEL_TEST_CASE_METHODS, ALL_DATA_TYPE_SEQ);

}  // namespace test

}  // namespace oneflow
