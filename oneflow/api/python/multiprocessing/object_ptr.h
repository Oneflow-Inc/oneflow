#pragma once

#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"

template<class T>
class THPPointer {
public:
  THPPointer(): ptr(nullptr) {};
  explicit THPPointer(T *ptr) noexcept : ptr(ptr) {};
  THPPointer(THPPointer &&p) noexcept { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~THPPointer() { free(); };
  T * get() { return ptr; }
  const T * get() const { return ptr; }
  T * release() { T *tmp = ptr; ptr = nullptr; return tmp; }
  operator T*() { return ptr; }
  THPPointer& operator =(T *new_ptr) noexcept { free(); ptr = new_ptr; return *this; }
  THPPointer& operator =(THPPointer &&p) noexcept { free(); ptr = p.ptr; p.ptr = nullptr; return *this; }
  T * operator ->() { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }

private:
  void free();
  T *ptr = nullptr;
};

/**
 * An RAII-style, owning pointer to a PyObject.  You must protect
 * destruction of this object with the GIL.
 *
 * WARNING: Think twice before putting this as a field in a C++
 * struct.  This class does NOT take out the GIL on destruction,
 * so if you will need to ensure that the destructor of your struct
 * is either (a) always invoked when the GIL is taken or (b) takes
 * out the GIL itself.  Easiest way to avoid this problem is to
 * not use THPPointer in this situation.
 */
using THPObjectPtr = THPPointer<PyObject>;
