#include "oneflow/core/kernel/pooling_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildPoolingKernel(const PoolingOpConf::PoolMethod& pooling_method) {
  OperatorConf op_conf;
  op_conf.set_name("pooling_test");
  PoolingOpConf* pooling_conf = op_conf.mutable_pooling_conf();
  pooling_conf->mutable_in()->set_name("pooling_in");
  pooling_conf->mutable_in()->set_data_type(GetDataType<T>::val);
  pooling_conf->mutable_out()->set_name("pooling_out");
  pooling_conf->mutable_out()->set_data_type(GetDataType<T>::val);
  pooling_conf->set_pool(pooling_method);
  pooling_conf->set_pad_h(1);
  pooling_conf->set_pad_w(1);
  pooling_conf->set_kernel_size_h(3);
  pooling_conf->set_kernel_size_w(3);
  pooling_conf->set_stride_h(2);
  pooling_conf->set_stride_w(2);

  auto pooling_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  pooling_op->ToProto(&op_proto);
  auto pooling_kernel = new PoolingKernel<device_type, T>();
  pooling_kernel->InitFromOpProto(op_proto);
  return pooling_kernel;
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    const PoolingOpConf::PoolMethod& pooling_method) {
  using KTC = KTCommon<device_type, T>;

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 5, 5}), GetDataType<T>::val, false);
  BlobDesc* idx_blob_desc =
      new BlobDesc(Shape({1, 1, 5, 5}), DataType::kUInt32, false);
  BlobDesc* out_blob_desc =
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false);
  (*bn2blob_ptr)["in"] = KTC::CreateBlobWithSpecifiedVal(
      in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  (*bn2blob_ptr)["out"] = KTC::CreateBlobWithRandomVal(out_blob_desc);
  (*bn2blob_ptr)["idx"] = KTC::CreateBlobWithRandomVal(idx_blob_desc);
  (*bn2blob_ptr)["out_diff"] = (*bn2blob_ptr)["out"];
  (*bn2blob_ptr)["in_diff"] = KTC::CreateBlobWithRandomVal(in_blob_desc);
  if (pooling_method == PoolingOpConf::kMax) {
    (*bn2blob_ptr)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
        out_blob_desc, {7, 9, 10, 17, 19, 20, 22, 24, 25});
    (*bn2blob_ptr)["expected_in_diff"] = KTC::CreateBlobWithSpecifiedVal(
        in_blob_desc, {0, 0, 0, 0,  0, 0,  7,  0, 9,  10, 0,  0, 0,
                       0, 0, 0, 17, 0, 19, 20, 0, 22, 0,  24, 25});
  } else if (pooling_method == PoolingOpConf::kAve) {
    (*bn2blob_ptr)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
        out_blob_desc, {16.0 / 9, 33.0 / 9, 28.0 / 9, 69.0 / 9, 13, 87.0 / 9,
                        76.0 / 9, 123.0 / 9, 88.0 / 9});
    (*bn2blob_ptr)["expected_in_diff"] = KTC::CreateBlobWithSpecifiedVal(
        in_blob_desc, {16.0 / 9 / 9,
                       16.0 / 9 / 9 + 33.0 / 9 / 9,
                       33.0 / 9 / 9,
                       33.0 / 9 / 9 + 28.0 / 9 / 9,
                       28.0 / 9 / 9,
                       16.0 / 9 / 9 + 69.0 / 9 / 9,
                       16.0 / 9 / 9 + 33.0 / 9 / 9 + 69.0 / 9 / 9 + 13.0 / 9,
                       33.0 / 9 / 9 + 13.0 / 9,
                       33.0 / 9 / 9 + 28.0 / 9 / 9 + 13.0 / 9 + 87.0 / 9 / 9,
                       28.0 / 9 / 9 + 87.0 / 9 / 9,
                       69.0 / 9 / 9,
                       69.0 / 9 / 9 + 13.0 / 9,
                       13.0 / 9,
                       13.0 / 9 + 87.0 / 9 / 9,
                       87.0 / 9 / 9,
                       69.0 / 9 / 9 + 76.0 / 9 / 9,
                       69.0 / 9 / 9 + 13.0 / 9 + 76.0 / 9 / 9 + 123.0 / 9 / 9,
                       13.0 / 9 + 123.0 / 9 / 9,
                       13.0 / 9 + 87.0 / 9 / 9 + 123.0 / 9 / 9 + 88.0 / 9 / 9,
                       87.0 / 9 / 9 + 88.0 / 9 / 9,
                       76.0 / 9 / 9,
                       76.0 / 9 / 9 + 123.0 / 9 / 9,
                       123.0 / 9 / 9,
                       123.0 / 9 / 9 + 88.0 / 9 / 9,
                       88.0 / 9 / 9});
  } else {
    TODO();
  }

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename T>
void TestPoolingKernel(const PoolingOpConf::PoolMethod& pooling_method) {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, T>(pooling_method);
  auto pooling_kernel = BuildPoolingKernel<device_type, T>(pooling_method);

  pooling_kernel->Forward(ctx, BnInOp2BlobPtr);
  pooling_kernel->Backward(ctx, BnInOp2BlobPtr);
  SyncStream<device_type>(&ctx);

  KTC::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTC::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
}

}  // namespace

}  // namespace test

TEST(PoolingKernel, pooling) {
#define POOLINGOPCONF (PoolingOpConf::kAve)(PoolingOpConf::kMax)
#define MAKE_ENTRY(device_type, data_pair, poolingop_conf)           \
  test::TestPoolingKernel<device_type, OF_PP_PAIR_FIRST(data_pair)>( \
      poolingop_conf);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, POOLINGOPCONF)
}

}  // namespace oneflow
