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

  //
  void EnrollWithPbnAndLbn(const std::string& pbn, const std::string& lbn);
  void EnrollWithLbn(const std::string& lbn);

  //
  Shape& MutPbnShape(const std::string& pbn) {
    return pbn2blob_desc_.at(pbn)->mut_shape();
  }
  Shape& MutLbnShape(const std::string& lbn) {
    return lbn2blob_desc_.at(lbn)->mut_shape();
  }

  const Shape& GetPbnShape(const std::string& pbn) const {
    return pbn2blob_desc_.at(pbn)->mut_shape();
  }
  virtual Shape GetLbnShape(const std::string& lbn) const {
    return lbn2blob_desc_.at(lbn)->mut_shape();
  }

 private:
  int32_t regst_desc_id_;
  const TaskNode* producer_;
  std::unordered_set<const TaskNode*> subscribers_;
  
  HashMap<std::string, std::unique_ptr<BlobDesc>> pbn2blob_desc_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> lbn2blob_desc_;

};

// Contiguous
class ContigRegstDesc final : public RegstDesc {
 public:
  static const char* kAllLbn;

  OF_DISALLOW_COPY_AND_MOVE(ContigRegstDesc);
  ContigRegstDesc() = default;
  ~ContigRegstDesc() = default;
  
  Shape GetLbnShape(const std::string& lbn) const override;

 private:
  Shape ComputeShape4AllLbn() const;

};

class DisContigRegstDesc final : public RegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DisContigRegstDesc);
  DisContigRegstDesc() = default;
  ~DisContigRegstDesc() = default;
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_REGISTER_DESC_H_
