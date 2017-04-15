#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

#include "common/util.h"
#include "common/shape.h"
#include "blob/blob_desc.h"

namespace oneflow {

class TaskNode;

// Regst : Register

class RegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDesc);
  RegstDesc();
  virtual ~RegstDesc() = default;

  //
  const TaskNode* GetProducer() const { return producer_; }
  void SetProducer(const TaskNode* task_node) { producer_ = task_node; }
  void AddSubscriber(const TaskNode* task_node) {
    CHECK(subscribers_.insert(task_node).second);
  }

  void CopyLbnAndShape(const RegstDesc*) { TODO(); }

  Shape* EnrollLbn(const std::string& lbn) { TODO(); }
  const Shape& GetShape(const std::string& lbn) { TODO(); }
  Shape* GetMutShapePtr(const std::string& lbn) { TODO(); }

 private:
  int32_t regst_desc_id_;
  const TaskNode* producer_;
  std::unordered_set<const TaskNode*> subscribers_;
  
  HashMap<std::string, std::unique_ptr<Shape>> lbn2shape_;

};

// Contiguous
class ContigRegstDesc final : public RegstDesc {
 public:
  static const char* kAllLbn;

  OF_DISALLOW_COPY_AND_MOVE(ContigRegstDesc);
  ContigRegstDesc() = default;
  ~ContigRegstDesc() = default;

 private:

};

class DisContigRegstDesc final : public RegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DisContigRegstDesc);
  DisContigRegstDesc() = default;
  ~DisContigRegstDesc() = default;
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_REGISTER_DESC_H_
