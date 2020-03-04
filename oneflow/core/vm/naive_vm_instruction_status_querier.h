#ifndef ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_

namespace oneflow {

class NaiveVmInstrStatusQuerier {
 public:
  ~NaiveVmInstrStatusQuerier() = default;

  bool done() const { return done_; }
  void set_done() { done_ = true; }

  static const NaiveVmInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const NaiveVmInstrStatusQuerier*>(mem_ptr);
  }
  static NaiveVmInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<NaiveVmInstrStatusQuerier*>(mem_ptr);
  }
  static NaiveVmInstrStatusQuerier* PlacementNew(char* mem_ptr) {
    return new (mem_ptr) NaiveVmInstrStatusQuerier();
  }

 private:
  NaiveVmInstrStatusQuerier(): done_(false) {}
  volatile bool done_;
};

}

#endif  // ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_
