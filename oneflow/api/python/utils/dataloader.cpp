/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef _WIN32

#include <atomic>
#include <map>
#include <set>
#include <csignal>
#include <sstream>
#include <sys/wait.h>

#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"

#include <stdexcept>

namespace oneflow {

namespace py = pybind11;

// reference: pytorch/torch/csrc/DataLoader.cpp
// https://github.com/pytorch/pytorch/blob/d69c22dd61a2f006dcfe1e3ea8468a3ecaf931aa/torch/csrc/DataLoader.cpp

// Critical signal handlers should be registered on worker processes before
// doing work.
// The handler will raise default handler so that the kill information will be
// retrieved from main process.
// Python handle is _set_worker_signal_handlers().
#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)                          \
  static void HANDLER_NAME(int sig, siginfo_t* info, void* ctx) {                \
    auto _w = write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char)); \
    (void)_w;                                                                    \
    struct sigaction sa {};                                                      \
    sa.sa_handler = SIG_DFL;                                                     \
    sa.sa_flags = 0;                                                             \
    if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGNAL, &sa, nullptr) != 0) { \
      _exit(EXIT_FAILURE);                                                       \
    } else {                                                                     \
      raise(SIGNAL);                                                             \
    }                                                                            \
  }

// signal(2) is really not portable. So use sigaction.
// http://man7.org/linux/man-pages/man2/signal.2.html
static inline void setSignalHandler(int signal, void (*handler)(int, siginfo_t*, void*),
                                    struct sigaction* old_sa_ptr) {
  struct sigaction sa {};
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(signal, &sa, old_sa_ptr) != 0) {
    std::ostringstream oss;
    oss << "An error occurred while setting handler for " << strsignal(signal) << ".";
    throw std::runtime_error(oss.str());
  }
}

SIGNAL_HANDLER(SIGBUS, handler_SIGBUS,
               "ERROR: Unexpected bus error encountered in worker. "
               "This might be caused by insufficient shared memory (shm).\n");
SIGNAL_HANDLER(SIGSEGV, handler_SIGSEGV,
               "ERROR: Unexpected segmentation fault encountered in worker.\n");
SIGNAL_HANDLER(SIGFPE, handler_SIGFPE,
               "ERROR: Unexpected floating-point exception encountered in worker.\n");

// When an error happened in DataLoader methods and Python starts to exit, the
// error trace will keep the loader alive, and Python may kill the children
// processes first before deleting the loader object. Then the cleaning up
// methods in DataLoader.__del__ are not yet called, and SIGCHILD will print an
// error saying a worker is killed by SIGTERM. So we suppress SIGTERM from main
// loader process here to avoid this by _exit(EXIT_SUCCESS). Note that if we
// exit with nonzero code, the loader SIGCHLD handler may report RuntimeError
// again, and then it defeats the whole purpose.
static void handler_SIGTERM(int sig, siginfo_t* info, void* ctx) {
  if (info->si_pid == getppid()) { _exit(EXIT_SUCCESS); }
  struct sigaction sa {};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGTERM, &sa, nullptr) != 0) {
    _exit(EXIT_FAILURE);
  } else {
    raise(SIGTERM);
  }
}

static void set_worker_signal_handlers() {
  setSignalHandler(SIGBUS, &handler_SIGBUS, nullptr);
  setSignalHandler(SIGSEGV, &handler_SIGSEGV, nullptr);
  setSignalHandler(SIGTERM, &handler_SIGTERM, nullptr);
  setSignalHandler(SIGFPE, &handler_SIGFPE, nullptr);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::map<int64_t, std::set<pid_t>> worker_pids = {};

static void error_if_any_worker_fails() {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int error;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::set<pid_t>* pid_set;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  pid_t worker_pid;
  siginfo_t infop;

  // Only check the pids we care about
  for (auto& w : worker_pids) {
    pid_set = &(w.second);
    for (auto pid_it = pid_set->begin(); pid_it != pid_set->end(); ++pid_it) {
      worker_pid = *pid_it;
      // Use waitid rather than waitpid so that we can set NOWAIT, and that Python
      // and other handlers can get whatever info they want about the child.
      infop.si_pid = 0;
      error = waitid(P_PID, worker_pid, &infop, WEXITED | WNOHANG | WNOWAIT);
      // ignore errors and case with no waitable child
      if (error < 0 || infop.si_pid == 0) continue;
      if (infop.si_code == CLD_EXITED && infop.si_status != EXIT_SUCCESS) {  // exit with error
        std::ostringstream oss;
        oss << "DataLoader worker (pid " << worker_pid << ") exited "
            << "unexpectedly with exit code " << infop.si_status << ". "
            << "Details are lost due to multiprocessing. Rerunning with "
            << "num_workers=0 may give better error trace.";
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set->clear();
        throw std::runtime_error(oss.str());
      } else if (infop.si_code == CLD_KILLED || infop.si_code == CLD_DUMPED) {  // killed by signal
        std::ostringstream oss;
        oss << "DataLoader worker (pid " << worker_pid << ") is killed "
            << "by signal: " << strsignal(infop.si_status) << ". ";
        if (infop.si_status == SIGBUS) {
          oss << "It is possible that dataloader's workers are out of shared memory. "
              << "Please try to raise your shared memory limit.";
        }
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set->clear();
        throw std::runtime_error(oss.str());
      }
    }
  }
}

inline int64_t utils_unpackLong(PyObject* obj) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int overflow;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) { throw py::value_error(); }
  if (overflow != 0) { throw std::runtime_error("Overflow when unpacking long"); }
  return (int64_t)value;
}

// We don't want to exit on any SIGCHLD from any child. child_pids is a tuple
// of pids we are interested in.
static void set_worker_pids(py::args py_args) {
  PyObject* args = py_args.ptr();
  if (PyTuple_GET_SIZE(args) != 2) {
    throw py::type_error("_set_worker_pids expects exactly 2 arguments.");
  }
  int64_t key = utils_unpackLong(PyTuple_GET_ITEM(args, 0));
  if (worker_pids.find(key) != worker_pids.end()) {
    throw py::value_error(
        "_set_worker_pids should be called only once for each _BaseDataLoaderIter.");
  }
  PyObject* child_pids = PyTuple_GET_ITEM(args, 1);
  if (!PyTuple_Check(child_pids)) {
    py::print("_set_worker_pids expects a tuple for child_pids, but got: ",
              Py_TYPE(child_pids)->tp_name);
    throw py::type_error("_set_worker_pids expects a tuple for child_pids");
  }

  std::set<pid_t> pids_set = {};
  auto size = PyTuple_GET_SIZE(child_pids);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    pids_set.insert(static_cast<pid_t>(utils_unpackLong(obj)));
  }

  worker_pids[key] = pids_set;
}

static void remove_worker_pids(py::args py_args) {
  PyObject* args = py_args.ptr();
  int64_t key = utils_unpackLong(PyTuple_GET_ITEM(args, 0));
  auto it = worker_pids.find(key);
  if (it == worker_pids.end()) {
    py::print("Cannot find worker information for _BaseDataLoaderIter with id :", key);
    throw py::value_error("Cannot find worker information for _BaseDataLoaderIter");
  }
  worker_pids.erase(it);
}

#undef SIGNAL_HANDLER

#else
// dummy implementations for windows

static PyObject* set_worker_signal_handlers(PyObject* module, PyObject* _ignored) {
  Py_RETURN_NONE;
}

static PyObject* set_worker_pids(PyObject* module, PyObject* _ignored) { Py_RETURN_NONE; }

static PyObject* remove_worker_pids(PyObject* module, PyObject* _ignored) { Py_RETURN_NONE; }

static PyObject* error_if_any_worker_fails(PyObject* module, PyObject* _ignored) { Py_RETURN_NONE; }

#endif

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("_set_worker_signal_handlers", &set_worker_signal_handlers);
  m.def("_set_worker_pids", &set_worker_pids);
  m.def("_remove_worker_pids", &remove_worker_pids);
  m.def("_error_if_any_worker_fails", &error_if_any_worker_fails);
}

}  // namespace oneflow
