#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/kernel/kernel.h"
#include <random>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {

#if defined(WITH_CUDA)

std::function<BlobDesc*(const std::string)> ConstructBn2BlobDescFunc(
    std::shared_ptr<Operator> op) {
  auto InsertBnsWithEmptyBlobDesc2Map =
      [](const std::vector<std::string>& bns,
         HashMap<std::string, BlobDesc*>* bn2blobdesc_map) {
        for (const std::string& bn : bns) {
          CHECK(bn2blobdesc_map->insert({bn, new BlobDesc}).second);
        }
      };
  auto bn2blobdesc_map = new HashMap<std::string, BlobDesc*>();
  InsertBnsWithEmptyBlobDesc2Map(op->data_tmp_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->input_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->input_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->output_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->output_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_tmp_bns(), bn2blobdesc_map);
  return [bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map->at(bn);
  };
}

template<>
Blob* CreateBlob<DeviceType::kCPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMallocHost(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(nullptr, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kCPU);
}

template<>
void BuildKernelCtx<DeviceType::kCPU>(KernelCtx* ctx) {
  ctx->device_ctx = new CpuDeviceCtx(-1);
}

template<>
void SyncStream<DeviceType::kCPU>(KernelCtx* ctx) {}

template<typename T>
class KTCommon<DeviceType::kCPU, T> final {
 public:
  static void CheckInitializeResult(const Blob* blob,
                                    const InitializerConf& initializer_conf) {
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
};

#define INSTANTIATE_KTCOMMON(type_cpp, type_proto) \
  template class KTCommon<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KTCOMMON, ALL_DATA_TYPE_SEQ)

#endif

namespace {

std::string ExpectedBlobName(const std::string& name) {
  return name + "_$expected$";
}

}  // namespace

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestCase<device_type>::CreateBlobWithSpecifiedVal(
    const BlobDesc* blob_desc, std::vector<T> val) const {
  //  return KTCommon<device_type, T>::CreateBlobWithSpecifiedValPtr(blob_desc,
  //                                                               &(val[0]));
  return CreateBlobWithSpecifiedValPtr(blob_desc, &(val[0]));
}

template<>
template<typename T>
Blob* OpKernelTestCase<DeviceType::kCPU>::CreateBlobWithSpecifiedValPtr(
    const BlobDesc* blob_desc, T* val) const {
  Blob* ret = CreateBlob<DeviceType::kCPU>(blob_desc);
  CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataContentField(),
                       cudaMemcpyHostToHost));
  return ret;
}

template<DeviceType device_type>
template<typename T>
Blob* OpKernelTestCase<device_type>::CreateBlobWithRandomVal(
    const BlobDesc* blob_desc) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 10);
  std::vector<T> val_vec(blob_desc->shape().elem_cnt());
  for (int64_t i = 0; i < blob_desc->shape().elem_cnt(); ++i) {
    val_vec[i] = static_cast<T>(dis(gen));
  }
  return CreateBlobWithSpecifiedVal<T>(blob_desc, val_vec);
}

template<DeviceType device_type>
template<typename T>
void OpKernelTestCase<device_type>::InitBlob(const std::string& name,
                                             const BlobDesc* blob_desc,
                                             const std::vector<T>& val) {
  Blob* blob = CreateBlobWithSpecifiedVal<T>(blob_desc, val);
  CHECK(bn_in_op2blob_.emplace(name, blob).second);
}

template<DeviceType device_type>
template<typename T>
void OpKernelTestCase<device_type>::ForwardCheckBlob(const std::string& name,

                                                     const BlobDesc* blob_desc,
                                                     const std::vector<T>& val,
                                                     bool need_random_init) {
  forward_asserted_blob_names_.push_back(name);
  if (need_random_init) {
    Blob* blob = CreateBlobWithRandomVal<T>(blob_desc);
    CHECK(bn_in_op2blob_.emplace(name, blob).second);
  }
  Blob* blob = CreateBlobWithSpecifiedVal<T>(blob_desc, val);
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
    Blob* blob = CreateBlobWithRandomVal<T>(blob_desc);
    CHECK(bn_in_op2blob_.emplace(name, blob).second);
  }
  Blob* blob = CreateBlobWithSpecifiedVal<T>(blob_desc, val);
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
    const BlobDesc* blob_desc) const {
  switch (blob_desc->data_type()) {
#define CREATE_BLOB_WITH_RANDOM_VAL_ENTRY(type_cpp, type_proto) \
  case type_proto: return CreateBlobWithRandomVal<type_cpp>(blob_desc);
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
      bn_in_op2blob_[bn_in_op] =
          SwitchCreateBlobWithRandomVal(&bn_in_op2blob_desc_.at(bn_in_op));
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
void OpKernelTestCase<device_type>::InitBeforeRun() {
  JobDescProto job_desc_proto;
  *job_desc_proto.mutable_job_conf() = job_conf_;
  if (Global<JobDesc>::Get()) { Global<JobDesc>::Delete(); }
  Global<JobDesc>::New(job_desc_proto);

  for (const auto& pair : bn_in_op2blob_) {
    bn_in_op2blob_desc_[pair.first] = pair.second->blob_desc();
  }

  BuildKernelCtx<device_type>(&kernel_ctx_);
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::set_is_train(bool is_train) {
  if (is_train) {
    mut_job_conf()->mutable_train_conf();
  } else {
    mut_job_conf()->mutable_predict_conf();
  }
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
                                                 const Blob* rhs) const {
  ASSERT_EQ(lhs->blob_desc(), rhs->blob_desc()) << blob_name;
  CHECK_EQ(lhs->data_type(), GetDataType<T>::val);
  if (IsFloatingPoint(lhs->data_type())) {
    for (int64_t i = 0; i < lhs->shape().elem_cnt(); ++i) {
      ASSERT_NEAR(lhs->dptr<T>()[i], rhs->dptr<T>()[i], 1e-6) << blob_name;
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
                                                  const Blob* rhs) const {
  switch (lhs->data_type()) {
#define BLOB_CMP_ENTRY(type_cpp, type_proto) \
  case type_proto: return BlobCmp<type_cpp>(blob_name, lhs, rhs);
    OF_PP_FOR_EACH_TUPLE(BLOB_CMP_ENTRY, ALL_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type>
void OpKernelTestCase<device_type>::Run() {
  InitBeforeRun();
  auto op = ConstructOp(op_conf_);
  auto BnInOp2BlobDesc = MakeGetterBnInOp2BlobDesc();
  op->InferBlobDescs(BnInOp2BlobDesc, &parallel_ctx_, device_type);
  std::list<bool> is_forward_launch_types;
  if (Global<JobDesc>::Get()->IsPredict() || is_forward_) {
    is_forward_launch_types = {true};
  } else {
    is_forward_launch_types = {true, false};
  }
  auto BnInOp2Blob = MakeGetterBnInOp2Blob();
  for (bool is_forward : is_forward_launch_types) {
    KernelConf kernel_conf;
    op->GenKernelConf(BnInOp2BlobDesc, is_forward, device_type, &parallel_ctx_,
                      &kernel_conf, nullptr);
    auto kernel = ConstructKernel(&parallel_ctx_, kernel_conf);
    kernel->Launch(kernel_ctx_, BnInOp2Blob);
  }
  SyncStream<device_type>(&kernel_ctx_);
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
  template void                                                              \
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
