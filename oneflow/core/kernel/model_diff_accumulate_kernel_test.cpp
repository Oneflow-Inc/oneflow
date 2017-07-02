#include "oneflow/core/kernel/model_diff_accumulate_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  Location loc;
  if (device_type == DeviceType::kCPU) {
    loc = Location::kHost;
  } else {
    loc = Location::kDevice;
  }

  std::vector<int64_t> dim_vec = {2, 4};
  FloatingPointType diff_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  FloatingPointType diff_acc_data[] = {5, 3, 2, 1, 7, 0, 1, 1};

  FloatingPointType expected_data[] = {6, 5, 5, 5, 12, 6, 8, 9};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  (*bn2blob_ptr)["model_diff"] =
      CreateBlobWithVector<FloatingPointType>(dim_vec, diff_data, loc);
  (*bn2blob_ptr)["model_diff_acc"] =
      CreateBlobWithVector<FloatingPointType>(dim_vec, diff_acc_data, loc);
  (*bn2blob_ptr)["expected_acc"] =
      CreateBlobWithVector<FloatingPointType>(dim_vec, expected_data, loc);
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildMdDiffAccKernel() {
  OperatorConf op_conf;
  op_conf.set_name("model_diff_acc");
  op_conf.mutable_model_diff_acc_conf();
  auto model_diff_acc_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  model_diff_acc_op->ToProto(&op_proto);

  auto model_diff_acc_kernel =
      new MdDiffAccKernel<device_type, FloatingPointType>();
  model_diff_acc_kernel->InitFromOpProto(op_proto);

  return model_diff_acc_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
void TestMdDiffAccKernel() {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();

  auto model_diff_acc_kernel =
      BuildMdDiffAccKernel<device_type, FloatingPointType>();

  model_diff_acc_kernel->Forward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);

  if (device_type == DeviceType::kCPU) {
    CheckResult<FloatingPointType>(BnInOp2BlobPtr, "model_diff_acc",
                                   "expected_acc",
                                   BlobCmpCpu<FloatingPointType>);
  } else {
    CheckResult<FloatingPointType>(BnInOp2BlobPtr, "model_diff_acc",
                                   "expected_acc",
                                   BlobCmpGpu<FloatingPointType>);
  }
}
}  // namespace test

TEST(MdDiffAccKernel, model_diff_acc_kernel_cpu) {
  test::TestMdDiffAccKernel<DeviceType::kCPU, float>();
  test::TestMdDiffAccKernel<DeviceType::kCPU, double>();
}

TEST(MdDiffAccKernel, model_diff_acc_kernel_gpu) {
  test::TestMdDiffAccKernel<DeviceType::kGPU, float>();
  test::TestMdDiffAccKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
