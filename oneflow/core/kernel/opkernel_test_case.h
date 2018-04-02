#ifndef ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
#define ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/kernel/kernel_context.h"

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
#else
#define TEST_GPU_OPKERNEL(func_name, data_type_seq, ...)
#endif

template<DeviceType device_type>
class OpKernelTestCase final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernelTestCase);
  OpKernelTestCase();
  ~OpKernelTestCase() = default;

  void Run();

  //  Setters
  void InitJobConf(const std::function<void(JobConf*)>& Init);
  void set_is_train(bool is_train);
  void set_is_forward(bool is_forward) { is_forward_ = is_forward; }
  OperatorConf* mut_op_conf() { return &op_conf_; }
  KernelCtx* mut_kernel_ctx() { return &kernel_ctx_; }
  HashMap<std::string, Blob*>* mut_bn_in_op2blob() { return &bn_in_op2blob_; }
  void set_initiation_before_backward(std::function<void()> Init) {
    initiation_before_backward_ = Init;
  }

  //  Getters
  const HashMap<std::string, Blob*>& bn_in_op2blob() const {
    return bn_in_op2blob_;
  }

  void EnrollBlobRegst(const std::string& blob_name, Regst*);
  template<typename T>
  Blob* InitBlob(const std::string&, const BlobDesc* blob_desc,
                 const std::vector<T>& val);
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
  static void BlobCmp(const std::string& blob_name, const Blob* lhs,
                      const Blob* rhs);

  template<typename T>
  static void CheckInitializeResult(const Blob* blob,
                                    const InitializerConf& initializer_conf);
  static Blob* CreateBlob(const BlobDesc*, Regst* regst);

 private:
  template<typename T>
  static Blob* CreateBlobWithRandomVal(const BlobDesc* blob_desc, Regst* regst);
  template<typename T>
  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc,
                                          std::vector<T> val, Regst* regst);
  template<typename T>
  static Blob* CreateBlobWithSpecifiedValPtr(const BlobDesc*, T* val,
                                             Regst* regst);
  static Blob* SwitchCreateBlobWithRandomVal(const BlobDesc* blob_desc,
                                             Regst* regst);
  static void SwitchBlobCmp(const std::string& blob_name, const Blob* lhs,
                            const Blob* rhs);
  static void SwitchCheckInitializeResult(
      const Blob* blob, const InitializerConf& initializer_conf);
  static void BuildKernelCtx(KernelCtx* ctx);
  static void SyncStream(KernelCtx* ctx);
  void UpdateGlobalJobDesc();

  std::function<Blob*(const std::string&)> MakeGetterBnInOp2Blob();
  std::function<BlobDesc*(const std::string&)> MakeGetterBnInOp2BlobDesc();
  void InitBeforeRun();
  void AssertAfterRun() const;

  HashMap<std::string, Blob*> bn_in_op2blob_;
  HashMap<std::string, BlobDesc> bn_in_op2blob_desc_;
  HashMap<std::string, Regst*> bn_in_op2regst_;
  JobConf job_conf_;
  OperatorConf op_conf_;
  std::list<std::string> forward_asserted_blob_names_;
  std::list<std::string> backward_asserted_blob_names_;
  ParallelContext parallel_ctx_;
  KernelCtx kernel_ctx_;
  bool is_forward_;
  std::function<void()> initiation_before_backward_;
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

#define MAKE_OPKERNEL_TEST_ENTRY(func_name, device_type_pair, data_type_pair, \
                                 ...)                                         \
  TEST(func_name,                                                             \
       OF_PP_JOIN(_, __COUNTER__, OF_PP_PAIR_FIRST(device_type_pair),         \
                  OF_PP_PAIR_FIRST(data_type_pair), ##__VA_ARGS__)) {         \
    OpKernelTestCase<OF_PP_PAIR_SECOND(device_type_pair)> opkernel_test_case; \
    func_name<OF_PP_PAIR_SECOND(device_type_pair),                            \
              OF_PP_PAIR_FIRST(data_type_pair)>                               \
        OF_PP_TUPLE_PUSH_FRONT(STRINGIZE_OPKERNEL_TEST_ARGS(__VA_ARGS__),     \
                               &opkernel_test_case);                          \
    opkernel_test_case.Run();                                                 \
  }

}  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
