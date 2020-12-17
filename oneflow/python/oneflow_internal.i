%module(directors="1") oneflow_internal
%include <std_string.i>
%include <std_pair.i>
%include <std_shared_ptr.i>
%include <stdint.i>
%include <typemaps.i>
%apply std::string *OUTPUT { std::string *error_str };
%include "oneflow/python/lib/core/Flat.i"
%include "oneflow/python/framework/oneflow_typemap.i"

%{
  
#include "oneflow/python/oneflow_internal.h"

%}
%feature("director:except") {
  if ($error != NULL) { LOG(FATAL) << "Swig::DirectorMethodException"; }
}

%include "oneflow/python/oneflow_internal.h"
