#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_PARSER_H_

#include "oneflow/customized/data/parser.h"
#include "oneflow/customized/data/coco_dataset.h"

namespace oneflow {
namespace data {

class COCOMeta;

class COCOParser final : public Parser<COCOImage> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<COCOImage>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  COCOParser(const std::shared_ptr<const COCOMeta>& meta) : meta_(meta){};
  ~COCOParser() = default;

  void Parse(std::shared_ptr<LoadTargetShdPtrVec> batch_data,
             user_op::KernelComputeContext* ctx) override;

 private:
  std::shared_ptr<const COCOMeta> meta_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_PARSER_H_
