#include "oneflow/core/operator/max_pooling_2d_op.h"
#include "oneflow/core/operator/average_pooling_2d_op.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
OpKernelTestCase* MaxPoolingTestCase(const std::string& job_type,
                                     const std::string& forward_or_backward) {
  OpKernelTestCase* pooling_test_case = new OpKernelTestCase();
  pooling_test_case->mut_job_conf()->set_default_data_type(GetDataType<T>::val);
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  pooling_test_case->set_device_type(device_type);
  ::oneflow::MaxPooling2DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_max_pooling_2d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(3);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(2);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  using KTC = KTCommon<device_type, T>;
  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 5, 5}), GetDataType<T>::val, false, false, 1);
  BlobDesc* out_blob_desc =
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false, false, 1);
  pooling_test_case->InitBlob(
      "in",
      KTC::CreateBlobWithSpecifiedVal(
          in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}));
  pooling_test_case->InitBlob(
      GenDiffBn("out"), KTC::CreateBlobWithSpecifiedVal(
                            out_blob_desc, {7, 9, 10, 17, 19, 20, 22, 24, 25}));
  pooling_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(out_blob_desc,
                                      {7, 9, 10, 17, 19, 20, 22, 24, 25}));
  pooling_test_case->BackwardCheckBlob(
      GenDiffBn("in"), device_type,
      KTC::CreateBlobWithSpecifiedVal(
          in_blob_desc, {0, 0, 0, 0,  0, 0,  7,  0, 9,  10, 0,  0, 0,
                         0, 0, 0, 17, 0, 19, 20, 0, 22, 0,  24, 25}));

  return pooling_test_case;
}

template<DeviceType device_type, typename T>
OpKernelTestCase* AveragePoolingTestCase(
    const std::string& job_type, const std::string& forward_or_backward) {
  OpKernelTestCase* pooling_test_case = new OpKernelTestCase();
  pooling_test_case->mut_job_conf()->set_default_data_type(GetDataType<T>::val);
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  pooling_test_case->set_device_type(device_type);
  ::oneflow::AveragePooling2DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_average_pooling_2d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(3);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(2);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  using KTC = KTCommon<device_type, T>;
  using KTC = KTCommon<device_type, T>;
  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 5, 5}), GetDataType<T>::val, false, false, 1);
  BlobDesc* out_blob_desc =
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false, false, 1);
  pooling_test_case->InitBlob(
      "in",
      KTC::CreateBlobWithSpecifiedVal(
          in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}));
  pooling_test_case->InitBlob(
      GenDiffBn("out"),
      KTC::CreateBlobWithSpecifiedVal(
          out_blob_desc, {16.0f / 4, 33.0f / 6, 28.0f / 4, 69.0f / 6, 13,
                          87.0f / 6, 76.0f / 4, 123.0f / 6, 88.0f / 4}));
  pooling_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(
          out_blob_desc, {16.0f / 4, 33.0f / 6, 28.0f / 4, 69.0f / 6, 13,
                          87.0f / 6, 76.0f / 4, 123.0f / 6, 88.0f / 4}));
  pooling_test_case->BackwardCheckBlob(
      GenDiffBn("in"), device_type,
      KTC::CreateBlobWithSpecifiedVal(
          in_blob_desc,
          {16.0f / 4 / 4,
           16.0f / 4 / 4 + 33.0f / 6 / 6,
           33.0f / 6 / 6,
           33.0f / 6 / 6 + 28.0f / 4 / 4,
           28.0f / 4 / 4,
           16.0f / 4 / 4 + 69.0f / 6 / 6,
           16.0f / 4 / 4 + 33.0f / 6 / 6 + 69.0f / 6 / 6 + 13.0f / 9,
           33.0f / 6 / 6 + 13.0f / 9,
           33.0f / 6 / 6 + 28.0f / 4 / 4 + 13.0f / 9 + 87.0f / 6 / 6,
           28.0f / 4 / 4 + 87.0f / 6 / 6,
           69.0f / 6 / 6,
           69.0f / 6 / 6 + 13.0f / 9,
           13.0f / 9,
           13.0f / 9 + 87.0f / 6 / 6,
           87.0f / 6 / 6,
           69.0f / 6 / 6 + 76.0f / 4 / 4,
           69.0f / 6 / 6 + 13.0f / 9 + 76.0f / 4 / 4 + 123.0f / 6 / 6,
           13.0f / 9 + 123.0f / 6 / 6,
           13.0f / 9 + 87.0f / 6 / 6 + 123.0f / 6 / 6 + 88.0f / 4 / 4,
           87.0f / 6 / 6 + 88.0f / 4 / 4,
           76.0f / 4 / 4,
           76.0f / 4 / 4 + 123.0f / 6 / 6,
           123.0f / 6 / 6,
           123.0f / 6 / 6 + 88.0f / 4 / 4,
           88.0f / 4 / 4}));

  return pooling_test_case;
}

TEST_CPU_ONLY_OPKERNEL(MaxPoolingTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward)(backward));

TEST_CPU_ONLY_OPKERNEL(AveragePoolingTestCase, FLOATING_DATA_TYPE_SEQ,
                       (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
