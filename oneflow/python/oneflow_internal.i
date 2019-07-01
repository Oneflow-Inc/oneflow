%module(directors="1") oneflow_internal
%include "oneflow/python/oneflow_typemap.i"
%{
#include "oneflow/python/oneflow_internal.h"
%}

%include <std_shared_ptr.i>
%shared_ptr(oneflow::ForeignCallback);
%feature("director") oneflow::ForeignCallback;
%include "oneflow/core/job/foreign_callback.h"
%include "oneflow/python/oneflow_internal.h"
