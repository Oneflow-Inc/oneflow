#ifndef ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_

namespace oneflow {
namespace vm {

class NaiveInstrStatusQuerier {
 public:
  ~NaiveInstrStatusQuerier() = default;

  bool done() const { return done_; }
  void set_done() { done_ = true; }

  static const NaiveInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const NaiveInstrStatusQuerier*>(mem_ptr);
  }
  static NaiveInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<NaiveInstrStatusQuerier*>(mem_ptr);
  }
  static NaiveInstrStatusQuerier* PlacementNew(char* mem_ptr) {
    return new (mem_ptr) NaiveInstrStatusQuerier();
  }

 private:
  NaiveInstrStatusQuerier() : done_(false) {}
  volatile bool done_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_
