#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/multiprocessing/object_ptr.h"
#include <csignal>

#include <stdexcept>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }


namespace oneflow {
namespace multiprocessing {

namespace py = pybind11;

void multiprocessing_init() {
  printf("\n================init.cpp >> multiprocessing_init================");
  auto multiprocessing_module =
      THPObjectPtr(PyImport_ImportModule("oneflow.multiprocessing"));
  if (!multiprocessing_module) {
    throw std::runtime_error("multiprocessing init error >> multiprocessing_module init fail!");
  }

  auto module = py::handle(multiprocessing_module).cast<py::module>();

  module.def("_prctl_pr_set_pdeathsig", [](int signal) {
#if defined(__linux__)
    auto rv = prctl(PR_SET_PDEATHSIG, signal);
    SYSASSERT(rv, "prctl");
#endif
  });


  printf("\n================init.cpp >> multiprocessing_init success!================\n");
  //Py_RETURN_TRUE;
}


ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("_multiprocessing_init", &multiprocessing_init);
}

} // namespace multiprocessing
} // namespace oneflow