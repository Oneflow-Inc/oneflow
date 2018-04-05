#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

namespace test {

namespace {

Regst* ConstructRegst(OpKernelTestCase* boxing_test_case,
                      DeviceType device_type) {
  boxing_test_case->InitJobConf([](JobConf* job_conf_ptr) {});

  static int64_t regst_desc_id = 0;
  RegstDescProto regst_desc_proto;
  regst_desc_proto.set_regst_desc_id(regst_desc_id++);
  regst_desc_proto.set_producer_task_id(0);
  regst_desc_proto.set_min_register_num(1);
  regst_desc_proto.set_max_register_num(1);
  regst_desc_proto.set_register_num(1);
  regst_desc_proto.mutable_mem_case()->mutable_host_pageable_mem();

  if (!Global<RegstMgr>::Get()) { Global<RegstMgr>::New(); }
  Regst* ret_regst = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(
      regst_desc_proto, device_type, RecordTypeProto::kOFRecord,
      [&ret_regst](Regst* regst) { ret_regst = regst; });

  return ret_regst;
}

}  // anonymous namespace

template<DeviceType device_type, typename T>
void BoxingConcatSplitTestCase(OpKernelTestCase* boxing_test_case,
                               const std::string& job_type,
                               const std::string& forward_or_backward) {
  const int32_t in_num = 4;
  const int32_t out_num = 2;
  boxing_test_case->set_is_train(job_type == "train");
  boxing_test_case->set_is_forward(forward_or_backward == "forward");
  BoxingOpConf* boxing_conf =
      boxing_test_case->mut_op_conf()->mutable_boxing_conf();
  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);
  boxing_conf->mutable_concat_box()->set_axis(1);
  BoxSplitConf* split_conf = boxing_conf->mutable_split_box();
  split_conf->set_axis(0);
  split_conf->add_part_num(2);
  split_conf->add_part_num(1);

  Regst* blob_regst = ConstructRegst(boxing_test_case, device_type);
  DataType data_type = GetDataType<T>::value;
  std::vector<Shape> blob_shape_in = {Shape({3, 1, 2, 1}), Shape({3, 2, 2, 1}),
                                      Shape({3, 3, 2, 1}), Shape({3, 4, 2, 1})};
  std::vector<Shape> blob_shape_out = {Shape({2, 10, 2, 1}),
                                       Shape({1, 10, 2, 1})};
  std::vector<BlobDesc*> blob_desc_in(4);
  std::vector<BlobDesc*> blob_desc_out(2);
  for (size_t i = 0; i < in_num; ++i) {
    boxing_test_case->EnrollBlobRegst("in_" + std::to_string(i), blob_regst);
    boxing_test_case->EnrollBlobRegst(GenDiffBn("in_" + std::to_string(i)),
                                      blob_regst);
    blob_desc_in[i] =
        new BlobDesc(blob_shape_in[i], data_type, false, false, 1);
    boxing_test_case->template InitBlob<T>(
        "in_" + std::to_string(i), blob_desc_in[i],
        std::vector<T>(blob_shape_in[i].elem_cnt(), i + 1));
  }
  for (size_t j = 0; j < out_num; ++j) {
    boxing_test_case->EnrollBlobRegst("out_" + std::to_string(j), blob_regst);
    blob_desc_out[j] =
        new BlobDesc(blob_shape_out[j], data_type, false, false, 1);
  }

  boxing_test_case->template ForwardCheckBlob<T>(
      "out_0", blob_desc_out[0],
      {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
       1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4});
  boxing_test_case->template ForwardCheckBlob<T>(
      "out_1", blob_desc_out[1],
      {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4});
}

TEST_CPU_ONLY_OPKERNEL(BoxingConcatSplitTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward));

template<DeviceType device_type, typename T>
void BoxingConcatCloneTestCase(OpKernelTestCase* boxing_test_case,
                               const std::string& job_type,
                               const std::string& forward_or_backward) {
  const int32_t in_num = 4;
  const int32_t out_num = 5;
  boxing_test_case->set_is_train(job_type == "train");
  boxing_test_case->set_is_forward(forward_or_backward == "forward");
  BoxingOpConf* boxing_conf =
      boxing_test_case->mut_op_conf()->mutable_boxing_conf();
  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);
  boxing_conf->mutable_concat_box()->set_axis(1);
  boxing_conf->mutable_clone_box();

  Regst* regst = ConstructRegst(boxing_test_case, device_type);
  DataType data_type = GetDataType<T>::value;
  std::vector<Shape> shape_in = {Shape({3, 4, 5, 5}), Shape({3, 2, 5, 5}),
                                 Shape({3, 1, 5, 5}), Shape({3, 7, 5, 5})};
  Shape shape_out({3, 14, 5, 5});
  for (size_t i = 0; i < in_num; ++i) {
    std::string bn = "in_" + std::to_string(i);
    int64_t elem_cnt = shape_in[i].elem_cnt();
    boxing_test_case->EnrollBlobRegst(bn, regst);
    BlobDesc* blob_desc = new BlobDesc(shape_in[i], data_type, false, false, 1);
    boxing_test_case->template InitBlob<T>(bn, blob_desc,
                                           std::vector<T>(elem_cnt, i + 1));
  }

  BlobDesc* blob_desc_out = new BlobDesc(shape_out, data_type, false, false, 1);
  std::vector<T> result;
  std::vector<T> piece;
  std::vector<T> seg_1(4 * 5 * 5, 1);
  std::vector<T> seg_2(2 * 5 * 5, 2);
  std::vector<T> seg_3(1 * 5 * 5, 3);
  std::vector<T> seg_4(7 * 5 * 5, 4);
  auto transformer = [](T t) -> T { return t; };
  std::transform(seg_1.begin(), seg_1.end(), std::back_inserter(piece),
                 transformer);
  std::transform(seg_2.begin(), seg_2.end(), std::back_inserter(piece),
                 transformer);
  std::transform(seg_3.begin(), seg_3.end(), std::back_inserter(piece),
                 transformer);
  std::transform(seg_4.begin(), seg_4.end(), std::back_inserter(piece),
                 transformer);
  std::copy(piece.begin(), piece.end(), std::back_inserter(result));
  std::copy(piece.begin(), piece.end(), std::back_inserter(result));
  std::copy(piece.begin(), piece.end(), std::back_inserter(result));
  for (size_t j = 0; j < out_num; ++j) {
    std::string bn = "out_" + std::to_string(j);
    boxing_test_case->EnrollBlobRegst(bn, regst);
    boxing_test_case->template ForwardCheckBlob<T>(bn, blob_desc_out, result);
  }
}

TEST_CPU_ONLY_OPKERNEL(BoxingConcatCloneTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward));

template<DeviceType device_type, typename T>
void BoxingAddCloneTestCase(OpKernelTestCase* boxing_test_case,
                            const std::string& job_type,
                            const std::string& forward_or_backward) {
  const int32_t in_num = 4;
  const int32_t out_num = 3;
  boxing_test_case->set_is_train(job_type == "train");
  boxing_test_case->set_is_forward(forward_or_backward == "forward");
  BoxingOpConf* boxing_conf =
      boxing_test_case->mut_op_conf()->mutable_boxing_conf();
  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);
  boxing_conf->mutable_add_box();
  boxing_conf->mutable_clone_box();

  DataType data_type = GetDataType<T>::value;
  Shape blob_shape({3, 4, 5, 5});
  BlobDesc* blob_desc = new BlobDesc(blob_shape, data_type, false, false, 1);
  size_t sum = 0;
  for (size_t i = 0; i < in_num; ++i) {
    std::string bn = "in_" + std::to_string(i);
    boxing_test_case->template InitBlob<T>(
        bn, blob_desc, std::vector<T>(blob_shape.elem_cnt(), i + 1));
    sum += i + 1;
  }
  for (size_t j = 0; j < out_num; ++j) {
    std::string bn = "out_" + std::to_string(j);
    boxing_test_case->template ForwardCheckBlob<T>(
        bn, blob_desc, std::vector<T>(blob_shape.elem_cnt(), sum));
  }
}

TEST_CPU_ONLY_OPKERNEL(BoxingAddCloneTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward));

template<DeviceType device_type, typename T>
void BoxingAddSplitTestCase(OpKernelTestCase* boxing_test_case,
                            const std::string& job_type,
                            const std::string& forward_or_backward) {
  const int32_t in_num = 4;
  const int32_t out_num = 2;
  boxing_test_case->set_is_train(job_type == "train");
  boxing_test_case->set_is_forward(forward_or_backward == "forward");
  BoxingOpConf* boxing_conf =
      boxing_test_case->mut_op_conf()->mutable_boxing_conf();
  boxing_conf->set_in_num(in_num);
  boxing_conf->set_out_num(out_num);
  boxing_conf->mutable_add_box();
  BoxSplitConf* split_conf = boxing_conf->mutable_split_box();
  split_conf->set_axis(1);
  split_conf->add_part_num(2);
  split_conf->add_part_num(2);

  Regst* regst = ConstructRegst(boxing_test_case, device_type);
  DataType data_type = GetDataType<T>::value;
  Shape shape_in({3, 4, 5, 5});
  std::vector<Shape> shape_out = {Shape({3, 2, 5, 5}), Shape({3, 2, 5, 5})};
  size_t sum = 0;
  for (size_t i = 0; i < in_num; ++i) {
    std::string bn = "in_" + std::to_string(i);
    boxing_test_case->EnrollBlobRegst(bn, regst);
    BlobDesc* blob_desc = new BlobDesc(shape_in, data_type, false, false, 1);
    boxing_test_case->template InitBlob<T>(
        bn, blob_desc, std::vector<T>(shape_in.elem_cnt(), i + 1));
    sum += i + 1;
  }
  boxing_test_case->EnrollBlobRegst("middle", regst);
  for (size_t j = 0; j < out_num; ++j) {
    std::string bn = "out_" + std::to_string(j);
    boxing_test_case->EnrollBlobRegst(bn, regst);
    BlobDesc* blob_desc =
        new BlobDesc(shape_out[j], data_type, false, false, 1);
    boxing_test_case->template ForwardCheckBlob<T>(
        bn, blob_desc, std::vector<T>(shape_out[j].elem_cnt(), sum));
  }
}

TEST_CPU_ONLY_OPKERNEL(BoxingAddSplitTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                       (train)(predict), (forward));

}  // namespace test

}  // namespace oneflow
