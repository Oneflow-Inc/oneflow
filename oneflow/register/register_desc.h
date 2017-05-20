#ifndef ONEFLOW_REGISTER_REGISTER_DESC_H_
#define ONEFLOW_REGISTER_REGISTER_DESC_H_

#include "common/util.h"
#include "common/shape.h"
#include "register/register_desc.pb.h"

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
  const HashSet<const TaskNode*>& subscribers() const { return subscribers_; }
  void AddSubscriber(const TaskNode*);

  // Lbn and Shape
  void CopyLbnFrom(const RegstDesc*);
  void CopyShapeFrom(const RegstDesc*);
  void EnrollLbn(const std::string& lbn);
  const Shape& GetShape(const std::string& lbn) const;
  Shape* GetMutShapePtr(const std::string& lbn);
  HashMap<std::string, std::unique_ptr<Shape>>& mut_lbn2shape();
  const HashMap<std::string, std::unique_ptr<Shape>>& lbn2shape() const;

  //
  void EraseZeroSizeBlob();
  int64_t CompElemCntOfAllBlob() const;
  std::string DebugStr() const;
  void ToProto(RegstDescProto*) const;
  
  static const char* kAllLbn;

 private:
  uint64_t regst_desc_id_;
  const TaskNode* producer_;
  HashSet<const TaskNode*> subscribers_;
  
  HashMap<std::string, std::unique_ptr<Shape>> lbn2shape_;
  int64_t register_num_;

};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_DESC_H_
