#ifndef ONEFLOW_GRAPH_REGISTER_DESC_H_
#define ONEFLOW_GRAPH_REGISTER_DESC_H_

namespace oneflow {

class RegisterDesc {
 public:
  void Add(const std::string& pbn) {}
  void Add(const std::string& pbn, const std::string& lbn) {}

  virtual void Init();

 private:
};

// Contiguous
class ContigRegistDesc : public RegisterDesc {
};

class DisContigRegistDesc : public RegisterDesc {
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_REGISTER_DESC_H_
