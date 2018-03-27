#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
OpKernelTestCase* ReluTestCase(const std::string& job_type,
                               const std::string& forward_or_backward) {
  OpKernelTestCase* relu_test_case = new OpKernelTestCase();
  relu_test_case->set_is_train(job_type == "train");
  relu_test_case->set_is_forward(forward_or_backward == "forward");
  relu_test_case->set_device_type(device_type);
  relu_test_case->mut_op_conf()->mutable_relu_conf();

  using KTC = KTCommon<device_type, T>;
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 8}), GetDataType<T>::val, false, false, 1);
  relu_test_case->InitBlob(
      "in", KTC::CreateBlobWithSpecifiedVal(blob_desc,
                                            {1, -1, -2, 2, 0, 5, -10, 100}));
  relu_test_case->InitBlob(
      GenDiffBn("out"),
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {-8, 7, -6, 5, -4, 3, -2, 1}));
  relu_test_case->ForwardCheckBlob(
      "out", device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {1, 0, 0, 2, 0, 5, 0, 100}));
  relu_test_case->BackwardCheckBlob(
      GenDiffBn("in"), device_type,
      KTC::CreateBlobWithSpecifiedVal(blob_desc, {-8, 0, 0, 5, 0, 3, 0, 1}));

  return relu_test_case;
}

TEST_CPU_ONLY_OPKERNEL(ReluTestCase,
                       FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ,
                       (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
