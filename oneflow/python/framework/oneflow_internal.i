%module(directors="1") oneflow_internal
%include <std_string.i>
%include <std_shared_ptr.i>
%include <stdint.i>
%include "Flat.i"
%include "oneflow/python/framework/oneflow_typemap.i"

%{
#define SWIG_FILE_WITH_INIT
#include "oneflow/python/framework/oneflow_internal.h"
%}

%shared_ptr(oneflow::ForeignCallback);
%feature("director") oneflow::ForeignCallback;
%include "oneflow/core/job/foreign_callback.h"
%include "oneflow/python/framework/oneflow_internal.h"
%template(CopyFromInt32Ndarry) CopyFromNdarry<int32_t>;
%template(CopyFromFloat32Ndarry) CopyFromNdarry<float>;
%template(CopyToInt32Ndarry) CopyToNdarry<int32_t>;
%template(CopyToFloat32Ndarry) CopyToNdarry<float>;
