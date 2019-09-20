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
#include "oneflow/python/oneflow_internal.e.h.expanded.h"
#include "oneflow/python/job_build_and_infer_if.h"

%}
%shared_ptr(oneflow::ForeignJobInstance);
%feature("director") oneflow::ForeignJobInstance;
%feature("director") oneflow::ForeignWatcher;
%feature("director:except") {
  if ($error != NULL) { LOG(FATAL) << "Swig::DirectorMethodException"; }
}
%include "oneflow/core/job/foreign_job_instance.h"
%include "oneflow/core/job/foreign_watcher.h"
%include "oneflow/python/oneflow_internal.h"
%include "oneflow/python/oneflow_internal.e.h.expanded.h"

%include "oneflow/python/job_build_and_infer_if.h"
