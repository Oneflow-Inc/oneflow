#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

#include "common/util.h"

namespace oneflow {

class ExecGraph;

class RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegisterDesc);
  RegisterDesc() {
    LOG(FATAL) << "TODO";
  }
  virtual ~RegisterDesc() = default;

  void Add(const std::string& pbn) {
    LOG(FATAL) << "TODO";
  }
  void Add(const std::string& pbn, const std::string& lbn) {
    LOG(FATAL) << "TODO";
  }

  void AddSubscriber(ExecGraph* exec_graph) {
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
