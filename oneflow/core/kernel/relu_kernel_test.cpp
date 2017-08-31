#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap() {
  using KTC = KTCommon<device_type, T>;

  auto bn2blob = new HashMap<std::string, Blob*>;
  BlobDesc* blob_desc = new BlobDesc(Shape({1, 8}), GetDataType<T>::val, false);

  (*bn2blob)["in"] = KTC::CreateBlobWithSpecifiedVal(
      blob_desc, {1, -1, -2, 2, 0, 5, -10, 100});
  (*bn2blob)["out"] = KTC::CreateBlobWithRandomVal(blob_desc);
  (*bn2blob)[GenDiffBn("in")] = KTC::CreateBlobWithRandomVal(blob_desc);
  (*bn2blob)[GenDiffBn("out")] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {-8, 7, -6, 5, -4, 3, -2, 1});
  (*bn2blob)["expected_out"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {1, 0, 0, 2, 0, 5, 0, 100});
  (*bn2blob)["expected_in_diff"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {-8, 0, 0, 5, 0, 3, 0, 1});
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildReluKernel() {
  DataType data_type = GetDataType<T>::val;
  OperatorConf op_conf;
  op_conf.set_name("relu_op_test");
  ReluOpConf* relu_conf = op_conf.mutable_relu_conf();
  relu_conf->mutable_in()->set_name("relu/in");
  relu_conf->mutable_in()->set_data_type(data_type);
  relu_conf->mutable_out()->set_name("relu/out");
  relu_conf->mutable_out()->set_data_type(data_type);
  auto relu_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  relu_op->ToProto(&op_proto);
  auto relu_kernel = new ReluKernel<device_type, T>();
  relu_kernel->InitFromOpProto(op_proto);
  return relu_kernel;
}

template<DeviceType device_type, typename T>
void TestReluKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2BlobPtr = BuildBnInOp2BlobMap<device_type, T>();
  auto relu_kernel = BuildReluKernel<device_type, T>();
  relu_kernel->Forward(ctx, BnInOp2BlobPtr);
  relu_kernel->Backward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);
  KTC::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTC::CheckResult(BnInOp2BlobPtr, GenDiffBn("in"), "expected_in_diff");
}

}  // namespace test

TEST(ReluKernel, relu_kernel_cpu) {
  test::TestReluKernel<DeviceType::kCPU, float>();
  test::TestReluKernel<DeviceType::kCPU, int32_t>();
}

TEST(ReluKernel, relu_kernel_gpu) {
  test::TestReluKernel<DeviceType::kGPU, float>();
  test::TestReluKernel<DeviceType::kGPU, int32_t>();
}

}  // namespace oneflow
