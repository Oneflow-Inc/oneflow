%module(directors="1") oneflow_internal
%include <std_string.i>
%include <std_shared_ptr.i>
%include <stdint.i>
%include "oneflow/python/lib/core/Flat.i"
%include "oneflow/python/framework/oneflow_typemap.i"

%{
  
#include "oneflow/python/framework/oneflow_internal.h"
#include "oneflow/python/framework/oneflow_internal.e.h.expanded.h"

%}

%shared_ptr(oneflow::ForeignCallback);
%feature("director") oneflow::ForeignCallback;
%include "oneflow/core/job/foreign_callback.h"
%include "oneflow/python/framework/oneflow_internal.h"
%include "oneflow/python/framework/oneflow_internal.e.h.expanded.h"
