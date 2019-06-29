%module oneflow_internal

%{
#include "oneflow/core/job/oneflow_internal.h"
%}

%include std_string.i
%inline %{
using namespace std;
%}

%typemap(in) const oneflow::JobSet& (oneflow::JobSet temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    PyErr_SetString(
        PyExc_TypeError,
        "Incorrect python bytes object");
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "Failed to parse serialized protobuf message");
    SWIG_fail;
  }
  $1 = &temp;
}

%include "oneflow/core/job/oneflow_internal.h"