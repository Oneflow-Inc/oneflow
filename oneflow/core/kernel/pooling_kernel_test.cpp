#include "oneflow/core/kernel/pooling_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPoinType>
Kernel* BuildPoolingKernel() {
  OperatorConf op_conf;
  op_conf.set_name("pooling_test");
  PoolingOpConf* pooling_conf = op_conf.mutable_pooling_conf();
  pooling_conf->set_in("pooling_in");
  pooling_conf->set_out("pooling_out");
  pooling_conf->set_pool(pooling_method);
  pooling_conf->add_pad(1);
  pooling_conf->add_kernel_size(3);
  pooling_conf->add_kernel_size(3);
  pooling_conf->add_stride(2);
  pooling_conf->add_stride(2);

  auto pooling_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  pooling_op->ToProto(&op_proto);
  auto pooling_kernel = new PoolingKernel<device_type, FloatingPointType>();
  pooling_kernel->InitFromOpProto(op_proto);
  return pooling_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[] = {};
  FloatingPointType out_mat[];
  FloatingPointType out_diff_mat[] = {};
  FloatingPointType in_diff_mat[];
  FloatingPointType expected_out_mat[] = {};
  FloatingPointType expected_in_diff_mat[] = {}

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({/**/}, in_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithVector({/**/}, out_mat);
  (*bn2blob_ptr)["out_diff"] = KTCommon::CreateBlobWithVector({/**/}, out_diff_mat);
  (*bn2blob_ptr)["in_diff"] = KTCommon::CreateBlobWithVector({/**/}, in_diff_mat);
  (*bn2blob_ptr)["expected_out"] = KTCommon::CreateBlobWithVector({/**/}, expected_out_mat);
  (*bn2blob_ptr)["expected_in_diff"] = KTCommon::CreateBlobWithVector({/**/}, expected_in_diff_mat);
  
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
void TestPoolingKernel() {
  using KTCOMMON = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto pooling_kernel = BuildPoolingKernel<device_type, FloatingPointType>();

  pooling_kernel->Forward(ctx, BnInOp2BlobPtr);
  pooling_kernel->Backward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
}

}  // namespace

}  // namespace test

TEST(PoolingKernel, pooling_cpu) {
  test::TestPoolingKernel<DeviceType::kCPU, float>();
  test::TestPoolingKernel<DeviceType::kCPU, double>();
}

TEST(PoolingKernel, pooling_gpu) {
  test::TestPoolingKernel<DeviceType::kGPU, float>();
  test::TestPoolingKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
