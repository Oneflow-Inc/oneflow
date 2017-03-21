#ifndef REFCOUNT_H_
#define REFCOUNT_H_

#include <atomic>

namespace oneflow {

namespace core {

class RefCounted {
 public:
  RefCounted();
  void Ref() const;
  bool Unref() const;
  bool RefCountIsOne() const;
 private:
  mutable std::atomic_int_fast32_t ref_; 
};
inline RefCounted::RefCounted() : ref_(1) {}

class ScopedUnref {
 public:
  explicit ScopedUnref(RefCounted* o) : obj_(o) {}
  ~ScopedUnref() {
    if(obj_) obj_->Unref();
  }
 private:
  RefCounted* obj_;
};

}

}

#endif
