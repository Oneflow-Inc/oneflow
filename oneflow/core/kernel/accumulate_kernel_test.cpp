#include "oneflow/core/kernel/accumulate_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildAccumulateKernel() {
  OperatorConf op_conf;
  op_conf.set_name("accumulate");
  op_conf.mutable_accumulate_conf();
  auto accumulate_op = ConstructOp(op_conf);

  OperatorProto op_proto;
  accumulate_op->ToProto(&op_proto);

  auto accumulate_kernel = new AccumulateKernel<device_type, T>();
  accumulate_kernel->InitFromOpProto(op_proto);

  return accumulate_kernel;
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap() {
  using KTC = KTCommon<device_type, T>;

  BlobDesc* blob_desc = new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false);

  auto bn2blob = new HashMap<std::string, Blob*>;

  (*bn2blob)["one"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {1, 2, 3, 4, 5, 6, 7, 8});
  (*bn2blob)["acc"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {5, 3, 2, 1, 7, 0, 1, 1});
  (*bn2blob)["expected_acc"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {6, 5, 5, 5, 12, 6, 8, 9});
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
void TestAccumulateKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobFunc = BuildBnInOp2BlobMap<device_type, T>();
  auto accumulate_kernel = BuildAccumulateKernel<device_type, T>();

  accumulate_kernel->Forward(ctx, BnInOp2BlobFunc);
  SyncStream<device_type>(&ctx);

  KTC::CheckResult(BnInOp2BlobFunc, "acc", "expected_acc");
}

}  // namespace

}  // namespace test

TEST(AccumulateKernel, accumulate) {
#define MAKE_ENTRY(x, y) test::TestAccumulateKernel<x, OF_PP_PAIR_FIRST(y)>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ)
}

}  // namespace oneflow
