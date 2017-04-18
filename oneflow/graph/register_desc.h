#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

#include "common/util.h"
#include "common/shape.h"

namespace oneflow {

// Regst  : Register
// Contig : Contiguous

class TaskNode;

class RegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDesc);
  RegstDesc();
  virtual ~RegstDesc() = default;

  // Producer
  const TaskNode* GetProducer() const { return producer_; }
  void SetProducer(const TaskNode* task_node) { producer_ = task_node; }

  // Lbn and Shape
  void CopyLbn2ShapeMap(const RegstDesc*);
  Shape* EnrollLbn(const std::string& lbn);
  const Shape& GetShape(const std::string& lbn);
  Shape* GetMutShapePtr(const std::string& lbn);
  
  static const char* kAllLbn;

 private:
  int32_t regst_desc_id_;
  const TaskNode* producer_;
  
  HashMap<std::string, std::unique_ptr<Shape>> lbn2shape_;

};

class ContigRegstDesc final : public RegstDesc {
 public:

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
