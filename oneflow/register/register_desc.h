#ifndef ONEFLOW_REGISTER_REGISTER_DESC_H_
#define ONEFLOW_REGISTER_REGISTER_DESC_H_

#include "common/util.h"
#include "common/shape.h"

namespace oneflow {

// Regst  : Register
// Contig : Contiguous

class TaskNode;

class RegstDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDesc);
  RegstDesc();
  ~RegstDesc() = default;

  // regst_desc_id
  uint64_t regst_desc_id() const { return regst_desc_id_; }
  void set_regst_desc_id(uint64_t val) { regst_desc_id_ = val; }
  // Producer
  const TaskNode* GetProducer() const { return producer_; }
  void SetProducer(const TaskNode* task_node) { producer_ = task_node; }

  // Lbn and Shape
  void CopyLbnFrom(const RegstDesc*);
  void CopyShapeFrom(const RegstDesc*);
  void EnrollLbn(const std::string& lbn);
  const Shape& GetShape(const std::string& lbn) const;
  Shape* GetMutShapePtr(const std::string& lbn);
  HashMap<std::string, std::unique_ptr<Shape>>& mut_lbn2shape() {
    return lbn2shape_;
  }

  static const char* kAllLbn;

 private:
  uint64_t regst_desc_id_;
  const TaskNode* producer_;
  
  HashMap<std::string, std::unique_ptr<Shape>> lbn2shape_;

};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_DESC_H_
