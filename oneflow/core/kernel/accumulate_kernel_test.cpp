#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void AccumulateKernelTestCase(OpKernelTestCase* test_case,
                              const std::string& job_type) {
  test_case->set_is_train(job_type == "train");
  test_case->set_is_forward(true);
  test_case->mut_op_conf()->mutable_accumulate_conf();
  BlobDesc* blob_desc =
      new BlobDesc(Shape({2, 4}), GetDataType<T>::value, false, false, 1);

  test_case->InitBlob<T>("one", blob_desc, {1, 2, 3, 4, 5, 6, 7, 8});
  test_case->InitBlob<T>("acc", blob_desc, {5, 3, 2, 1, 7, 0, 1, 1});
  test_case->ForwardCheckBlob<T>("acc", blob_desc, {6, 5, 5, 5, 12, 6, 8, 9},
                                 false);
}

TEST_CPU_AND_GPU_OPKERNEL(AccumulateKernelTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict));

}  // namespace test

}  // namespace oneflow
