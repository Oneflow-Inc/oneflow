#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

#include "common/util.h"

namespace oneflow {

class TaskNode;

class RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegisterDesc);
  RegisterDesc() {
    LOG(FATAL) << "TODO";
  }
  virtual ~RegisterDesc() = default;

  // Pbn means that no other task need it
  // Lbn means that there are other tasks who need this blob
  void AddPbn(const std::string& pbn) {
    LOG(FATAL) << "TODO";
  }
  void AddLbn(const std::string& lbn) {
    LOG(FATAL) << "TODO";
  }

  void AddSubscriber(TaskNode* task_node) {
    LOG(FATAL) << "TODO";
  }

 private:

};

// Contiguous
class ContigRegistDesc final : public RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ContigRegistDesc);
  ContigRegistDesc() {
    LOG(FATAL) << "TODO";
  }
  ~ContigRegistDesc() = default;

};

class DisContigRegistDesc final : public RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DisContigRegistDesc);
  DisContigRegistDesc() {
    LOG(FATAL) << "TODO";
  }
  ~DisContigRegistDesc() = default;
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_REGISTER_DESC_H_
