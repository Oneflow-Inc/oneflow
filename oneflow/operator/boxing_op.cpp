#include "operator/boxing_op.h"
#include <string>
#include <vector>
#include "operator/operator_manager.h"
#include "common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_boxing_conf());
  mut_op_conf() = op_conf;

  for (int64_t i = 0; i < op_conf.boxing_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
  }
  EnrollDataTmpBn("middle");
  for (int64_t i = 0; i < op_conf.boxing_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

const PbMessage& BoxingOp::GetSpecialConf() const {
  return op_conf().boxing_conf();
}

std::string BoxingOp::ibn2lbn(const std::string& input_bn) const {
  return GetStringFromSpecialConf("lbn");
}

std::string BoxingOp::obn2lbn(const std::string& output_bn) const {
  return GetStringFromSpecialConf("lbn");
}

void BoxingOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  auto boxing_conf = op_conf().boxing_conf();
  // concat input blob shape to middle blob shape
  int32_t concat_axis = boxing_conf.concat_box().axis();
  CHECK(concat_axis == 0 || concat_axis == 1);
  std::vector<int64_t> data_tmp_blob_shape_vec =
    GetShapePtr4BnInOp(input_bns().at(0))->dim_vec();
  for (size_t ib_idx = 1; ib_idx < input_bns().size(); ++ib_idx) {
    auto ib_shape_vec = GetShapePtr4BnInOp(input_bns().at(ib_idx))->dim_vec();
    for (size_t i = 0; i < ib_shape_vec.size(); ++i) {
      if (i == concat_axis) {
        data_tmp_blob_shape_vec[i] += ib_shape_vec[i];
      } else {
        CHECK_EQ(data_tmp_blob_shape_vec[i], ib_shape_vec[i]);
      }
    }
  }
  CHECK_EQ(data_tmp_bns().size(), 1);
  *GetShapePtr4BnInOp(data_tmp_bns()[0]) = Shape(data_tmp_blob_shape_vec);
  auto out_box_case = boxing_conf.out_box_case();
  CHECK_NE(out_box_case, BoxingOpConf::OUT_BOX_NOT_SET);
  if (out_box_case == BoxingOpConf::kDataSplitBox) {
    uint32_t out_num = output_bns().size();
    BalancedSplitter splitter(data_tmp_blob_shape_vec[0], out_num);
    auto output_shape_vec = data_tmp_blob_shape_vec;
    for (size_t i = 0; i < out_num; ++i) {
      output_shape_vec[0] = splitter.At(i).size();
      *GetShapePtr4BnInOp(output_bns()[i]) = Shape(output_shape_vec);
    }
  } else if (out_box_case == BoxingOpConf::kCloneBox) {
    for (auto obn : output_bns()) {
      *GetShapePtr4BnInOp(obn) = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNEXPECTED_RUN();
  }
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
