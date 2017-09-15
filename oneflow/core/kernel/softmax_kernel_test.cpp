#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap() {
  using KTC = KTCommon<device_type, T>;

  DataType data_type = GetDataType<T>::val;
  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["in"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false), {1, 2, 3, 4, 0, 0, 0, 0});
  (*bn2blob)["out"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2, 4}), data_type, false));
  (*bn2blob)["tmp"] =
      KTC::CreateBlobWithRandomVal(new BlobDesc(Shape({2}), data_type, false));
  (*bn2blob)["in_diff"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2, 4}), data_type, false));
  (*bn2blob)["out_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false),
      {0.2f, 1, 2, 3, -4, 3, -2, 1});
  (*bn2blob)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false),
      {0.0320586f, 0.0871443f, 0.2368828f, 0.6439143f, 0.25f, 0.25f, 0.25f,
       0.25f});
  (*bn2blob)["expected_in_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false),
      {-0.0737048f, -0.1306350f, -0.1182198f, 0.3225595f, -0.875f, 0.875f,
       -0.375f, 0.375f});
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildSoftmaxKernel() {
  OperatorConf op_conf;
  op_conf.set_name("softmax_op_test");
  SoftmaxOpConf* softmax_conf = op_conf.mutable_softmax_conf();
  softmax_conf->set_in("softmax/in");
  softmax_conf->set_out("softmax/out");
  auto softmax_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  softmax_op->ToProto(&op_proto);
  auto softmax_kernel = new SoftmaxKernel<device_type, T>();
  softmax_kernel->InitFromOpProto(op_proto);
  return softmax_kernel;
}

template<DeviceType device_type, typename T>
void TestSoftmaxKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto bn2blob = BuildBnInOp2BlobMap<device_type, T>();
  auto softmax_kernel = BuildSoftmaxKernel<device_type, T>();
  softmax_kernel->Forward(ctx, bn2blob);
  softmax_kernel->Backward(ctx, bn2blob);
  SyncStream<device_type>(&ctx);
  KTC::CheckResult(bn2blob, "out", "expected_out");
  KTC::CheckResult(bn2blob, "in_diff", "expected_in_diff");
}

}  // namespace

}  // namespace test

TEST(SoftmaxKernel, softmax) {
#define MAKE_ENTRY(device_type, data_type_pair) \
  test::TestSoftmaxKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ)
}

}  // namespace oneflow
