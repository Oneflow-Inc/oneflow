#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

#include "common/util.h"
#include "common/shape.h"

namespace oneflow {

class TaskNode;

class RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegisterDesc);
  RegisterDesc();
  virtual ~RegisterDesc() = default;

  //
  const TaskNode* GetProducer() const { return producer_; }
  void SetProducer(const TaskNode* task_node) { producer_ = task_node; }
  void AddSubscriber(const TaskNode* task_node) {
    CHECK(subscribers_.insert(task_node).second);
  }

  //
  void AddPbn(const std::string& pbn) {
    CHECK(pbn2shape_.emplace(pbn, Shape()).second);
  }
  void AddLbn(const std::string& lbn) {
    CHECK(lbn2shape_.emplace(lbn, Shape()).second);
  }

  //
  Shape& MutPbnShape(const std::string& pbn) {
    return pbn2shape_.at(pbn);
  }
  Shape& MutLbnShape(const std::string& lbn) {
    return lbn2shape_.at(lbn);
  }
  const Shape& GetPbnShape(const std::string& pbn) const {
    return pbn2shape_.at(pbn);
  }
  virtual Shape GetLbnShape(const std::string& lbn) const {
    return lbn2shape_.at(lbn);
  }

 private:
  int32_t register_desc_id_;
  const TaskNode* producer_;
  std::unordered_set<const TaskNode*> subscribers_;
  // Pbn means that no other task need it
  // Lbn means that there are other tasks who need this blob
  HashMap<std::string, Shape> pbn2shape_;
  HashMap<std::string, Shape> lbn2shape_;

};

// Contiguous
class ContigRegistDesc final : public RegisterDesc {
 public:
  static const char* kLogicalAllBlobName;

  OF_DISALLOW_COPY_AND_MOVE(ContigRegistDesc);
  ContigRegistDesc() = default;
  ~ContigRegistDesc() = default;
  
  Shape GetLbnShape(const std::string& lbn) const override {
    if (lbn == kLogicalAllBlobName) {
      return ComputeShape4AllLbn();
    } else {
      return RegisterDesc::GetLbnShape(lbn);
    }
  }
 private:
  Shape ComputeShape4AllLbn() const;

};

class DisContigRegistDesc final : public RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DisContigRegistDesc);
  DisContigRegistDesc() = default;
  ~DisContigRegistDesc() = default;
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_REGISTER_DESC_H_
