#include "oneflow/core/kernel/clone_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildCloneKernel(int out_num) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  CloneOpConf* clone_conf = op_conf.mutable_clone_conf();
  clone_conf->set_out_num(out_num);
  clone_conf->set_lbn("clone_kernel_test");
  auto clone_op = OpMgr::Singleton()->ConstructOp(op_conf);
  OperatorProto op_proto;
  clone_op->ToProto(&op_proto);
  auto clone_kernel = new CloneKernel<device_type, FloatingPointType>();
  clone_kernel->InitFromOpProto(op_proto);
  return clone_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(int out_num) {
  std::vector<int64_t> dim_vec = {1, 3, 2};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] =
      KernelTestCommon<device_type, FloatingPointType>::CreateBlobWithSameValue(
          dim_vec, 1);
  (*bn2blob_ptr)["in_diff"] =
      KernelTestCommon<device_type, FloatingPointType>::CreateBlobWithSameValue(
          dim_vec, 2);
  (*bn2blob_ptr)["in_diff_expected"] =
      KernelTestCommon<device_type, FloatingPointType>::CreateBlobWithSameValue(
          dim_vec, 4 * out_num);
  for (size_t i = 0; i != out_num; ++i) {
    (*bn2blob_ptr)["out_" + std::to_string(i)] =
        KernelTestCommon<device_type,
                         FloatingPointType>::CreateBlobWithSameValue(dim_vec,
                                                                     3);
    (*bn2blob_ptr)["out_" + std::to_string(i) + "_diff"] =
        KernelTestCommon<device_type,
                         FloatingPointType>::CreateBlobWithSameValue(dim_vec,
                                                                     4);
  }
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestCloneKernel() {
  KernelCtx ctx;
  KernelTestCommon<device_type, FloatingPointType>::BuildKernelCtx(&ctx);

  int out_num = 3;
  auto BnInOp2BlobPtr =
      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(out_num);
  auto clone_kernel = BuildCloneKernel<device_type, FloatingPointType>(out_num);

  clone_kernel->Forward(ctx, BnInOp2BlobPtr);
  clone_kernel->Backward(ctx, BnInOp2BlobPtr);
  KernelTestCommon<device_type, FloatingPointType>::SyncStream(&ctx);

  for (size_t i = 0; i != out_num; ++i) {
    KernelTestCommon<device_type, FloatingPointType>::CheckResult(
        BnInOp2BlobPtr, "in", "out_" + std::to_string(i));
  }
  KernelTestCommon<device_type, FloatingPointType>::CheckResult(
      BnInOp2BlobPtr, "in_diff", "in_diff_expected");
}

}  // namespace test

TEST(CloneKernel, clone_cpu) {
  test::TestCloneKernel<DeviceType::kCPU, float>();
  test::TestCloneKernel<DeviceType::kCPU, double>();
}

TEST(CloneKernel, clone_gpu) {
  test::TestCloneKernel<DeviceType::kGPU, float>();
  test::TestCloneKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
