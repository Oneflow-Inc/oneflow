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

#define TEST_CPU_ONLY_OPKERNEL(func_name, data_type_seq, ...)                \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_OPKERNEL_TEST_ENTRY, (func_name),    \
                                   ((cpu, DeviceType::kCPU)), data_type_seq, \
                                   __VA_ARGS__)

#define TEST_CPU_AND_GPU_OPKERNEL(func_name, data_type_seq, ...)         \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(                                      \
      MAKE_OPKERNEL_TEST_ENTRY, (func_name),                             \
      ((cpu, DeviceType::kCPU))((gpu, DeviceType::kGPU)), data_type_seq, \
      __VA_ARGS__)

class OpKernelTestCase final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernelTestCase);
  OpKernelTestCase();
  ~OpKernelTestCase() = default;

  void Run();

  //  Setters
  JobConf* mut_job_conf() { return &job_conf_; }
  void set_is_train(bool is_train);
  void set_device_type(DeviceType device_type) { device_type_ = device_type; }
  OperatorConf* mut_op_conf() { return &op_conf_; }
  KernelCtx* mut_kernel_ctx() { return &kernel_ctx_; }
  void InitBlob(const std::string&, Blob* blob);
  void ForwardCheckBlob(const std::string&, DeviceType device_type, Blob* blob);
  void ForwardCheckBlob(const std::string&, DeviceType device_type, Blob* blob,
                        bool need_random_init);
  void BackwardCheckBlob(const std::string&, DeviceType device_type, Blob* blob,
                         bool need_random_init);
  void BackwardCheckBlob(const std::string&, DeviceType device_type,
                         Blob* blob);
  void set_is_forward(bool is_forward) { is_forward_ = is_forward; }

 private:
  std::function<Blob*(const std::string&)> MakeGetterBnInOp2Blob();
  std::function<BlobDesc*(const std::string&)> MakeGetterBnInOp2BlobDesc();
  void InitBeforeRun();
  void AssertAfterRun() const;

  HashMap<std::string, Blob*> bn_in_op2blob_;
  HashMap<std::string, BlobDesc> bn_in_op2blob_desc_;
  JobConf job_conf_;
  OperatorConf op_conf_;
  std::list<std::string> forward_asserted_blob_names_;
  std::list<std::string> backward_asserted_blob_names_;
  ParallelContext parallel_ctx_;
  KernelCtx kernel_ctx_;
  DeviceType device_type_;
  bool is_forward_;
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
    func_name<OF_PP_PAIR_SECOND(device_type_pair),                            \
              OF_PP_PAIR_FIRST(data_type_pair)>                               \
        STRINGIZE_OPKERNEL_TEST_ARGS(__VA_ARGS__)->Run();                     \
  }

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_CASE_H_
