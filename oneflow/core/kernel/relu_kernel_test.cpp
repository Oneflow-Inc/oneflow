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
  OperatorConf op_conf;
  op_conf.set_name("relu_op_test");
  ReluOpConf* relu_conf = op_conf.mutable_relu_conf();
  relu_conf->set_in("relu/in");
  relu_conf->set_out("relu/out");
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
  auto BnInOp2Blob = BuildBnInOp2BlobMap<device_type, T>();
  auto relu_kernel = BuildReluKernel<device_type, T>();
  relu_kernel->Forward(ctx, BnInOp2Blob);
  relu_kernel->Backward(ctx, BnInOp2Blob);
  SyncStream<device_type>(&ctx);
  KTC::CheckResult(BnInOp2Blob, "out", "expected_out");
  KTC::CheckResult(BnInOp2Blob, GenDiffBn("in"), "expected_in_diff");
}

}  // namespace test

TEST(ReluKernel, relu) {
#define MAKE_ENTRY(device_type, data_type_pair) \
  test::TestReluKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
      MAKE_ENTRY, DEVICE_TYPE_SEQ,
      FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ)
}

}  // namespace oneflow
