#ifndef REFCOUNT_H_
#define REFCOUNT_H_

namespace oneflow {

namespace core {

class RefCounted {
 public:
  RefCounted();
  void Ref() const;
  bool Unref() const;
  bool RefCountIsOne() const;
 
};

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
