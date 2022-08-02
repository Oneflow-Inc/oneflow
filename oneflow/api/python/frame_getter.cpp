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
#include "oneflow/core/common/frame_getter.h"

#include <Python.h>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/common/foreign_lock_helper.h"

namespace py = pybind11;

namespace oneflow {

using FrameVector = small_vector<std::pair<PyFrameObject*, int>, 10>;

class PyFrameGetter final : public FrameGetter {
 public:
  void RecordCurrentFrame(int64_t id) override {
    if (IsShuttingDown()) { return; }
    id %= frames_.max_size();
    frames_[id].clear();
    PyFrameObject* frame = PyEval_GetFrame();
    CHECK_NOTNULL(frame);
    Py_INCREF(frame);
    while (frame != nullptr) {
      int lineno = PyFrame_GetLineNumber(frame);
      frames_[id].push_back(std::make_pair(frame, lineno));
      frame = frame->f_back;
    }
  }
  void Print(int64_t id) const override {
    id %= frames_.max_size();
    CHECK_JUST(Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
      if (frames_[id].empty()) { return Maybe<void>::Ok(); }
      FrameVector frame_and_no_vec = frames_.at(id);  // NOLINT
      for (auto it = frame_and_no_vec.rbegin(); it != frame_and_no_vec.rend(); ++it) {
        const auto& frame_and_no = *it;
        PyFrameObject* frame = frame_and_no.first;
        int lineno = frame_and_no.second;

        const std::string filename = PyBytes_AS_STRING(
            PyUnicode_AsEncodedString(frame->f_code->co_filename, "utf-8", "~E~"));
        const std::string funcname =
            PyBytes_AS_STRING(PyUnicode_AsEncodedString(frame->f_code->co_name, "utf-8", "~E~"));
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
          std::cout << "??? " << filename << std::endl;
          return Maybe<void>::Ok();
        }
        std::string line_text;
        for (int j = 0; j < lineno; ++j) {
          std::getline(ifs, line_text);
        }
        line_text.erase(line_text.find_last_not_of(' ')+1);         //suffixing spaces
        line_text.erase(0, line_text.find_first_not_of(' '));       //prefixing spaces
        std::cout << "  File \"" << filename << "\", line " << lineno << ", in " << funcname << std::endl;
        std::cout << "    " << line_text << std::endl;
      }
      return Maybe<void>::Ok();
    }));
  }

 private:
  std::array<FrameVector, 5000> frames_;
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("RegisterFrameGetter", []() {
    Singleton<FrameGetter>::Delete();
    Singleton<FrameGetter>::SetAllocated(new PyFrameGetter());
  });
  m.def("PrintFrame", [](int64_t id) { Singleton<FrameGetter>::Get()->Print(id); });
}

}  // namespace oneflow
