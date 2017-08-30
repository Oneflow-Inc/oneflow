#include "oneflow/core/kernel/accumulate_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTC = KTCommon<device_type, T>;

  std::vector<int64_t> dim_vec = {2, 4};
  BlobDesc* blob_desc = new BlobDesc;
  blob_desc->set_data_type(GetDataType<T>::val);
  blob_desc->mut_shape() = Shape(dim_vec);

  T diff_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  T diff_acc_data[] = {5, 3, 2, 1, 7, 0, 1, 1};

  T expected_data[] = {6, 5, 5, 5, 12, 6, 8, 9};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  (*bn2blob_ptr)["one"] = KTC::CreateBlobWithSpecifiedVal(blob_desc, diff_data);

  (*bn2blob_ptr)["acc"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, diff_acc_data);
  (*bn2blob_ptr)["expected_acc"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc, expected_data);
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildAccumulateKernel() {
  OperatorConf op_conf;
  op_conf.set_name("model_diff_acc");
  op_conf.mutable_accumulate_conf();
  auto model_diff_acc_op = ConstructOp(op_conf);

  OperatorProto op_proto;
  model_diff_acc_op->ToProto(&op_proto);

  auto model_diff_acc_kernel = new AccumulateKernel<device_type, T>();
  model_diff_acc_kernel->InitFromOpProto(op_proto);

  return model_diff_acc_kernel;
}

template<DeviceType device_type, typename T>
void TestAccumulateKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, T>();

  auto model_diff_acc_kernel = BuildAccumulateKernel<device_type, T>();

  model_diff_acc_kernel->Forward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);

  KTC::CheckResult(BnInOp2BlobPtr, "acc", "expected_acc");
}

}  // namespace

}  // namespace test

TEST(AccumulateKernel, accumulate) {
#define MAKE_ENTRY(x, y) test::TestAccumulateKernel<x, OF_PP_FIRST_ARG y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ)
}

}  // namespace oneflow
