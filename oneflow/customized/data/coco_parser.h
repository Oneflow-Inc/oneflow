#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_PARSER_H_

#include "oneflow/customized/data/parser.h"
#include "oneflow/customized/data/coco_dataset.h"

namespace oneflow {

class COCOMeta;

class COCOParser final : public Parser<COCOImage> {
 public:
  using LoadTargetPtr = std::shared_ptr<COCOImage>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;

  COCOParser(COCOMeta* meta) : meta_(meta){};
  ~COCOParser() = default;

  void Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
             user_op::KernelComputeContext* ctx) override;

 private:
  const COCOMeta* meta_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_PARSER_H_
