#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

#include "common/util.h"

namespace oneflow {

class TransfmGraph;

class RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegisterDesc);
  RegisterDesc() = default;
  ~RegisterDesc() = default;

  void Add(const std::string& pbn) {
    LOG(FATAL) << "TODO";
  }
  void Add(const std::string& pbn, const std::string& lbn) {
    LOG(FATAL) << "TODO";
  }

  void AddSubscriber(TransfmGraph* transfm_graph) {
    LOG(FATAL) << "TODO";
  }

  virtual void Init() {
    LOG(FATAL) << "TODO";
  }

 private:

};

// Contiguous
class ContigRegistDesc final : public RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ContigRegistDesc);
  ContigRegistDesc() = default;
  ~ContigRegistDesc() = default;

  void Init() override {
    LOG(FATAL) << "TODO";
  }

};

class DisContigRegistDesc final : public RegisterDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DisContigRegistDesc);
  DisContigRegistDesc() = default;
  ~DisContigRegistDesc() = default;
  
  void Init() override {
    LOG(FATAL) << "TODO";
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_REGISTER_DESC_H_
