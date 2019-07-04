%module(directors="1") oneflow_internal
%include <std_string.i>
%include <std_shared_ptr.i>
%include "oneflow/python/framework/oneflow_typemap.i"

%{
#include "oneflow/python/framework/oneflow_internal.h"
%}

%shared_ptr(oneflow::ForeignCallback);
%feature("director") oneflow::ForeignCallback;
%include "oneflow/core/job/foreign_callback.h"
%include "oneflow/python/framework/oneflow_internal.h"
