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
#ifndef ONEFLOW_CORE_COMMON_STACKTRACE_H_
#define ONEFLOW_CORE_COMMON_STACKTRACE_H_

#include <filesystem>

#include "oneflow/core/common/error.h"
#define private protected
#include "oneflow/extension/stack/stacktrace.h"
#undef private

namespace oneflow {

using namespace backward;

class StackPrinter : protected Printer {
 public:
  StackPrinter() {
    inliner_context_size = 1;
    trace_context_size = 1;
  }

 protected:
  bool is_oneflow_file(const std::string& filename) {
    return std::string(std::filesystem::path(filename).filename()).find("oneflow")
           != std::string::npos;
  }

  template<typename ST>
  void print_stacktrace(ST& st, std::ostream& os, Colorize& colorize) {
    print_header(os, st.thread_id());
    _resolver.load_stacktrace(st);
    if (reverse) {
      for (size_t trace_idx = st.size(); trace_idx > 0; --trace_idx) {
        print_trace(os, _resolver.resolve(st[trace_idx - 1]), colorize);
      }
    } else {
      for (size_t trace_idx = 0; trace_idx < st.size(); ++trace_idx) {
        print_trace(os, _resolver.resolve(st[trace_idx]), colorize);
      }
    }
    // Add a new line before Python stack
    os << std::endl;
  }

  void print_trace(std::ostream& os, const ResolvedTrace& trace, Colorize& colorize) {
    if (!is_oneflow_file(trace.object_filename)) { return; }
    // os << "#" << std::left << std::setw(2) << trace.idx << std::right;
    bool already_indented = true;

    if (!trace.source.filename.size() || object) {
      os << "   Object \"" << trace.object_filename << "\", at " << trace.addr << ", in "
         << trace.object_function << "\n";
      // already_indented = false;
    }

    for (size_t inliner_idx = trace.inliners.size(); inliner_idx > 0; --inliner_idx) {
      if (!already_indented) { os << "   "; }
      const ResolvedTrace::SourceLoc& inliner_loc = trace.inliners[inliner_idx - 1];
      print_source_loc(os, " | ", inliner_loc);
      if (snippet) {
        print_snippet(os, "   ", inliner_loc, colorize, Color::purple, inliner_context_size);
      }
      // already_indented = false;
    }

    if (trace.source.filename.size()) {
      if (!already_indented) { os << "   "; }
      print_source_loc(os, "  ", trace.source, trace.addr);
      if (snippet) {
        print_snippet(os, "   ", trace.source, colorize, Color::yellow, trace_context_size);
      }
    }
  }

  void print_snippet(std::ostream& os, const char* indent,
                     const ResolvedTrace::SourceLoc& source_loc, Colorize& colorize,
                     Color::type color_code, int context_size) {
    using namespace std;
    typedef SnippetFactory::lines_t lines_t;

    lines_t lines = _snippets.get_snippet(source_loc.filename, source_loc.line,
                                          static_cast<unsigned>(context_size));

    for (lines_t::const_iterator it = lines.begin(); it != lines.end(); ++it) {
      if (it->first == source_loc.line) {
        colorize.set_color(color_code);
        if (lines.size() > 1) {
          os << indent << ">";
        } else {
          os << indent << " ";
        }
      } else {
        os << indent << " ";
      }
      const auto pos = it->second.find_first_not_of(" \t");
      os << std::setw(4) << it->second.substr(pos, it->second.size() - pos) << "\n";
      if (it->first == source_loc.line) { colorize.set_color(Color::reset); }
    }
  }

  void print_source_loc(std::ostream& os, const char* indent,
                        const ResolvedTrace::SourceLoc& source_loc, void* addr = nullptr) {
    os << "  File \"" << source_loc.filename << "\", line " << source_loc.line << ", in "
       << source_loc.function;

    if (address && addr != nullptr) { os << " [" << addr << "]"; }
    os << "\n";
  }

 public:
  template<typename ST>
  FILE* print(ST& st, FILE* fp = stderr) {
    backward::cfile_streambuf obuf(fp);
    std::ostream os(&obuf);
    backward::Colorize colorize(os);
    colorize.activate(color_mode, fp);
    print_stacktrace(st, os, colorize);
    return fp;
  }

  // template<typename ST>
  // FILE* print(ST& st, FILE* fp = stderr) {
  //   // cfile_streambuf obuf(fp);
  //   // std::ostream os(&obuf);
  //   // Colorize colorize(os);
  //   // colorize.activate(color_mode, fp);
  //   // print_stacktrace(st, os, colorize);
  //   std::cout << std::endl;
  //   return fp;
  // }
  // void print_trace(std::ostream& os, const ResolvedTrace& trace, Colorize& colorize) {
  // }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STACKTRACE_H_
