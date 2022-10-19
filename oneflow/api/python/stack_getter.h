#ifndef ONEFLOW_API_PYTHON_STACK_GETTER_H_
#define ONEFLOW_API_PYTHON_STACK_GETTER_H_

// Used by oneflow/api/python/custom_eval_frame.c
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <frameobject.h>

void push_frame(PyFrameObject* frame);
void pop_frame();

#ifdef __cplusplus
}
#endif

#endif  // ONEFLOW_API_PYTHON_STACK_GETTER_H_
