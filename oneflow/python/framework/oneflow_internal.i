%module(directors="1") oneflow_internal
%include <std_string.i>
%include <std_shared_ptr.i>
%include "oneflow/python/framework/oneflow_typemap.i"
%include "Flat.i"

%{
#define SWIG_FILE_WITH_INIT
#include "oneflow/python/framework/oneflow_internal.h"
%}

%shared_ptr(oneflow::ForeignCallback);
%feature("director") oneflow::ForeignCallback;
%include "oneflow/core/job/foreign_callback.h"
%include "oneflow/python/framework/oneflow_internal.h"
%template(CopyFromIntNdarry) CopyFromNdarry<int>;
%template(CopyFromFloatNdarry) CopyFromNdarry<float>;
%template(CopyToIntNdarry) CopyToNdarry<int>;
%template(CopyToFloatNdarry) CopyToNdarry<float>;
