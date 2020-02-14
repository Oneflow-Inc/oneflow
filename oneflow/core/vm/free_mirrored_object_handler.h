#ifndef ONEFLOW_CORE_VM_FREE_MIRRORED_OBJECT_HANDLER_H_
#define ONEFLOW_CORE_VM_FREE_MIRRORED_OBJECT_HANDLER_H_

namespace oneflow {

class LogicalObject;

class FreeMirroredObjectHandler {
 public:
  virtual ~FreeMirroredObjectHandler() = default;

  virtual void Call(LogicalObject*) const = 0;

 protected:
  FreeMirroredObjectHandler() = default;
};

class FreeMirroredObjectIgnoreHandler : public FreeMirroredObjectHandler {
 public:
  ~FreeMirroredObjectIgnoreHandler() override = default;

  void Call(LogicalObject*) const override {}

  static const FreeMirroredObjectIgnoreHandler* Singleton() {
    static const FreeMirroredObjectIgnoreHandler singleton;
    return &singleton;
  }

 private:
  FreeMirroredObjectIgnoreHandler() : FreeMirroredObjectHandler() {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_FREE_MIRRORED_OBJECT_HANDLER_H_
