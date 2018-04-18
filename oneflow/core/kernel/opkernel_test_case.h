#ifndef ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
#define ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_

#include <random>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {

#define TEST_CPU_ONLY_OPKERNEL TEST_CPU_OPKERNEL
#define TEST_GPU_ONLY_OPKERNEL TEST_GPU_OPKERNEL
#define TEST_CPU_AND_GPU_OPKERNEL(func_name, data_type_seq, ...) \
  TEST_CPU_OPKERNEL(func_name, data_type_seq, __VA_ARGS__)       \
  TEST_GPU_OPKERNEL(func_name, data_type_seq, __VA_ARGS__)

#define TEST_CPU_OPKERNEL(func_name, data_type_seq, ...)                     \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_OPKERNEL_TEST_ENTRY, (func_name),    \
                                   ((cpu, DeviceType::kCPU)), data_type_seq, \
                                   __VA_ARGS__)

#ifdef WITH_CUDA
#define TEST_GPU_OPKERNEL(func_name, data_type_seq, ...)                     \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_OPKERNEL_TEST_ENTRY, (func_name),    \
                                   ((gpu, DeviceType::kGPU)), data_type_seq, \
                                   __VA_ARGS__)
#define TEST_DIFF_KERNEL_IMPL(func_name, ...)                               \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_TEST_DIFF_KERNEL_IMPL, (func_name), \
                                   ##__VA_ARGS__)
#else
#define TEST_GPU_OPKERNEL(func_name, data_type_seq, ...)
#define TEST_DIFF_KERNEL_IMPL(func_name, ...)
#endif

inline std::string ExpectedBlobName(const std::string& name) {
  return name + "_$expected$";
}

DataType DataType4CppTypeString(const std::string& cpp_type_str);

class OpKernelTestCase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernelTestCase);
  OpKernelTestCase();
  virtual ~OpKernelTestCase() = default;

  //  Setters
  void InitJobConf(const std::function<void(JobConf*)>& Init);
  void set_is_train(bool is_train);
  void set_is_forward(bool is_forward) { is_forward_ = is_forward; }
  void set_default_device_type(DeviceType default_device_type) {
    default_device_type_ = default_device_type;
  }
  void set_default_data_type(DataType default_data_type);
  OperatorConf* mut_op_conf() { return &op_conf_; }
  KernelCtx* mut_kernel_ctx() { return &kernel_ctx_; }
  HashMap<std::string, Blob*>* mut_bn_in_op2blob() { return &bn_in_op2blob_; }
  void set_initiation_before_backward(std::function<void()> Init) {
    initiation_before_backward_ = Init;
  }
  void SetBlobSpecializedDeviceType(const std::string& blob_name,
                                    DeviceType dev_type) {
    bn_in_op2device_type_.emplace(blob_name, dev_type);
  }
  BlobDesc* MutBlobDesc4BnInOp(const std::string& bn_in_op) {
    return &bn_in_op2blob_desc_[bn_in_op];
  }

  //  Getters
  DeviceType default_device_type() const { return default_device_type_; }
  DataType default_data_type() const {
    return Global<JobDesc>::Get()->DefaultDataType();
  }
  const HashMap<std::string, Blob*>& bn_in_op2blob() const {
    return bn_in_op2blob_;
  }
  DeviceType GetBlobDeviceType(const std::string& blob_name) const {
    const auto& it = bn_in_op2device_type_.find(blob_name);
    return (it == bn_in_op2device_type_.end()) ? default_device_type()
                                               : it->second;
  }
  const BlobDesc* BlobDesc4BnInOp(const std::string& bn_in_op) const {
    const auto& it = bn_in_op2blob_desc_.find(bn_in_op);
    return it == bn_in_op2blob_desc_.end() ? nullptr : &it->second;
  }

  void EnrollBlobRegst(const std::string& blob_name, Regst*);
  template<typename T>
  Blob* InitBlob(const std::string&, const BlobDesc* blob_desc,
                 const std::vector<T>& val);
  template<typename T>
  Blob* RandomInitBlob(const std::string&, const BlobDesc* blob_desc);
  template<typename T>
  void ForwardCheckBlob(const std::string&, const BlobDesc* blob_desc,
                        const std::vector<T>& val);
  template<typename T>
  void BackwardCheckBlob(const std::string&, const BlobDesc* blob_desc,
                         const std::vector<T>& val);

  template<typename T>
  void ForwardCheckBlob(const std::string&, const BlobDesc* blob_desc,
                        const std::vector<T>& val, bool need_random_init);
  template<typename T>
  void BackwardCheckBlob(const std::string&, const BlobDesc* blob_desc,
                         const std::vector<T>& val, bool need_random_init);

  template<typename T>
  void ForwardCheckBlobWithAnother(
      const std::string&, const BlobDesc* blob_desc,
      const std::string& expected_existed_blob_name, bool need_random_init);
  template<typename T>
  void BackwardCheckBlobWithAnother(
      const std::string&, const BlobDesc* blob_desc,
      const std::string& expected_existed_blob_name, bool need_random_init);

  // usually, you should not call it
  void Run();

 protected:
  void InitBeforeRun();
  void InferBlobDesc(std::shared_ptr<Operator>* op, OpContext** op_context);
  void RunKernel(Operator* op, OpContext* op_context);
  void AssertAfterRun() const;
  Regst* GetBlobRegst(const std::string& bn_in_op);
  bool is_forward() const { return is_forward_; }
  std::function<Blob*(const std::string&)> MakeGetterBnInOp2Blob();

 private:
  template<typename T>
  void CheckBlob(const std::string&, const BlobDesc* blob_desc,
                 const std::vector<T>& val, bool need_random_init);

  template<typename T>
  void CheckBlobWithAnother(const std::string&, const BlobDesc* blob_desc,
                            const std::string& expected_existed_blob_name,
                            bool need_random_init);

  void UpdateGlobalJobDesc();

  std::function<BlobDesc*(const std::string&)> MakeGetterBnInOp2BlobDesc();

  bool is_forward_;
  DeviceType default_device_type_;
  std::function<void()> initiation_before_backward_;
  ParallelContext parallel_ctx_;
  KernelCtx kernel_ctx_;
  JobConf job_conf_;
  OperatorConf op_conf_;
  HashMap<std::string, Blob*> bn_in_op2blob_;
  HashMap<std::string, BlobDesc> bn_in_op2blob_desc_;
  HashMap<std::string, Regst*> bn_in_op2regst_;
  HashMap<std::string, DeviceType> bn_in_op2device_type_;
  std::list<std::string> forward_asserted_blob_names_;
  std::list<std::string> backward_asserted_blob_names_;
};

// diff results runned by differnt kernel implementation
class DiffKernelImplTestCase final : public OpKernelTestCase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DiffKernelImplTestCase);
  DiffKernelImplTestCase(bool is_train, bool is_forward,
                         DataType default_data_type);
  ~DiffKernelImplTestCase() = default;

  void SetBlobNames(
      const std::list<std::string>& input_or_weight_bn_in_op,
      const std::list<std::string>& output_bn_in_op,
      const std::list<std::string>& output_diff_bn_in_op,
      const std::list<std::string>& input_or_weight_diff_bn_in_op);
  void SetInputBlobDesc(const std::string& bns_in_op, const Shape& shape,
                        DataType data_type);

  void set_initiate_kernel_ctx(
      const std::function<
          void(const std::function<Blob*(const std::string&)>&)>&
          initate_kernel_ctx) {
    initiate_kernel_ctx_ = initate_kernel_ctx;
  }

  // usually, you should not call it
  void MultiRunThenCheck();

 private:
  void RandomInitInputOrigin();
  void InitInputBlobs();
  void CopyBlobDesc4DiffBlob();
  void DumpBlobs(const std::string& prefix);
  void CheckMultiRunResults(const std::string& base_prefix,
                            const std::list<std::string>& other_prefixes) const;
  std::list<std::string> AllInputBlobNames() const;
  std::list<std::string> AllOutputBlobNamesWithValidBlob() const;
  static std::string GetOriginInputBlobName(const std::string& bn_in_op) {
    return std::string("origin_") + bn_in_op;
  }

  std::list<std::string> input_blob_names_;
  std::list<std::string> output_blob_names_;
  std::list<std::string> input_diff_blob_names_;
  std::list<std::string> output_diff_blob_names_;
  std::function<void(const std::function<Blob*(const std::string&)>&)>
      initiate_kernel_ctx_;
};

template<DeviceType device_type>
struct OpKernelTestUtil final {
  template<typename T>
  static void BlobCmp(const std::string& blob_name, const Blob* lhs,
                      DeviceType lhs_device_type, const Blob* rhs,
                      DeviceType rhs_device_type);
  template<typename T>
  static Blob* CreateBlobWithRandomVal(const BlobDesc* blob_desc, Regst* regst);
  template<typename T>
  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc,
                                          std::vector<T> val, Regst* regst);
  template<typename T>
  static void CheckInitializeResult(const Blob* blob,
                                    const InitializerConf& initializer_conf);

  static Blob* CreateBlob(const BlobDesc*, Regst* regst);
  static void BuildKernelCtx(KernelCtx* ctx);
  static void SyncStream(KernelCtx* ctx);

 private:
  template<typename T>
  static Blob* CreateBlobWithSpecifiedValPtr(const BlobDesc*, T* val,
                                             Regst* regst);
};

#define STRINGIZE_OPKERNEL_TEST_ARGS(...)            \
  OF_PP_CAT(OF_PP_CAT(STRINGIZE_OPKERNEL_TEST_ARGS_, \
                      OF_PP_VARIADIC_SIZE(__VA_ARGS__))(__VA_ARGS__), )
#define STRINGIZE_OPKERNEL_TEST_ARGS_0() ()
#define STRINGIZE_OPKERNEL_TEST_ARGS_1(a0) (#a0)
#define STRINGIZE_OPKERNEL_TEST_ARGS_2(a0, a1) (#a0, #a1)
#define STRINGIZE_OPKERNEL_TEST_ARGS_3(a0, a1, a2) (#a0, #a1, #a2)
#define STRINGIZE_OPKERNEL_TEST_ARGS_4(a0, a1, a2, a3) (#a0, #a1, #a2, #a3)
#define STRINGIZE_OPKERNEL_TEST_ARGS_5(a0, a1, a2, a3, a4) \
  (#a0, #a1, #a2, #a3, #a4)
#define STRINGIZE_OPKERNEL_TEST_ARGS_6(a0, a1, a2, a3, a4, a5) \
  (#a0, #a1, #a2, #a3, #a4, #a5)
#define STRINGIZE_OPKERNEL_TEST_ARGS_7(a0, a1, a2, a3, a4, a5, a6) \
  (#a0, #a1, #a2, #a3, #a4, #a5, #a6)

#define MAKE_OPKERNEL_TEST_ENTRY(func_name, device_type_pair, data_type_pair, \
                                 ...)                                         \
  TEST(func_name,                                                             \
       OF_PP_JOIN(_, __COUNTER__, OF_PP_PAIR_FIRST(device_type_pair),         \
                  OF_PP_PAIR_FIRST(data_type_pair), ##__VA_ARGS__)) {         \
    OpKernelTestCase opkernel_test_case;                                      \
    opkernel_test_case.set_default_device_type(                               \
        OF_PP_PAIR_SECOND(device_type_pair));                                 \
    func_name<OF_PP_PAIR_SECOND(device_type_pair),                            \
              OF_PP_PAIR_FIRST(data_type_pair)>                               \
        OF_PP_TUPLE_PUSH_FRONT(STRINGIZE_OPKERNEL_TEST_ARGS(__VA_ARGS__),     \
                               &opkernel_test_case);                          \
    opkernel_test_case.Run();                                                 \
  }

#define MAKE_TEST_DIFF_KERNEL_IMPL(func_name, ...)                         \
  TEST(func_name, OF_PP_JOIN(_, __COUNTER__, ##__VA_ARGS__)) {             \
    auto* test_case = func_name STRINGIZE_OPKERNEL_TEST_ARGS(__VA_ARGS__); \
    test_case->MultiRunThenCheck();                                        \
  }

}  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
