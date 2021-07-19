#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/thread/thread_unique_tag.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
	m.def("SetThisThreadUniqueTag", [](const std::string& thread_tag) {
		return SetThisThreadUniqueTag(thread_tag).GetOrThrow();
	});
}

}
