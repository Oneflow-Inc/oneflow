#include "oneflow/api/python/multiprocessing/object_ptr.h"


template<>
void THPPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<PyObject>;