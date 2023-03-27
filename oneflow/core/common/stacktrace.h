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

    if (!trace.source.filename.size() || object) {
      os << "   Object \"" << trace.object_filename << "\", at " << trace.addr << ", in "
         << trace.object_function << "\n";
    }

    for (size_t inliner_idx = trace.inliners.size(); inliner_idx > 0; --inliner_idx) {
      const ResolvedTrace::SourceLoc& inliner_loc = trace.inliners[inliner_idx - 1];
      print_source_loc(os, " | ", inliner_loc);
      if (snippet) {
        print_snippet(os, "   ", inliner_loc, colorize, Color::purple, inliner_context_size);
      }
    }

    if (trace.source.filename.size()) {
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
};

class OneFlowSignalHandling : protected backward::SignalHandling {
 public:
  explicit OneFlowSignalHandling(const std::vector<int>& posix_signals = make_default_signals())
      : _loaded(false) {
    bool success = true;

    const size_t stack_size = 1024 * 1024 * 8;
    _stack_content.reset(static_cast<char*>(malloc(stack_size)));
    if (_stack_content) {
      stack_t ss;
      ss.ss_sp = _stack_content.get();
      ss.ss_size = stack_size;
      ss.ss_flags = 0;
      if (sigaltstack(&ss, nullptr) < 0) { success = false; }
    } else {
      success = false;
    }

    for (size_t i = 0; i < posix_signals.size(); ++i) {
      struct sigaction action;
      memset(&action, 0, sizeof action);
      action.sa_flags = static_cast<int>(SA_SIGINFO | SA_ONSTACK | SA_NODEFER | SA_RESETHAND);
      sigfillset(&action.sa_mask);
      sigdelset(&action.sa_mask, posix_signals[i]);
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#endif
      action.sa_sigaction = &sig_handler;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

      int r = sigaction(posix_signals[i], &action, nullptr);
      if (r < 0) success = false;
    }

    _loaded = success;
  };

  static void handleSignal(int, siginfo_t* info, void* _ctx) {
    ucontext_t* uctx = static_cast<ucontext_t*>(_ctx);

    StackTrace st;
    void* error_addr = nullptr;
#ifdef REG_RIP  // x86_64
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.gregs[REG_RIP]);
#elif defined(REG_EIP)  // x86_32
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.gregs[REG_EIP]);
#elif defined(__arm__)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.arm_pc);
#elif defined(__aarch64__)
#if defined(__APPLE__)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext->__ss.__pc);
#else
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.pc);
#endif
#elif defined(__mips__)
    error_addr =
        reinterpret_cast<void*>(reinterpret_cast<struct sigcontext*>(&uctx->uc_mcontext)->sc_pc);
#elif defined(__ppc__) || defined(__powerpc) || defined(__powerpc__) || defined(__POWERPC__)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.regs->nip);
#elif defined(__riscv)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.__gregs[REG_PC]);
#elif defined(__s390x__)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext.psw.addr);
#elif defined(__APPLE__) && defined(__x86_64__)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext->__ss.__rip);
#elif defined(__APPLE__)
    error_addr = reinterpret_cast<void*>(uctx->uc_mcontext->__ss.__eip);
#else
#warning ":/ sorry, ain't know no nothing none not of your architecture!"
#endif
    if (error_addr) {
      st.load_from(error_addr, 32, reinterpret_cast<void*>(uctx), info->si_addr);
    } else {
      st.load_here(32, reinterpret_cast<void*>(uctx), info->si_addr);
    }

    StackPrinter printer;
    printer.print(st, stderr);

#if (defined(_XOPEN_SOURCE) && _XOPEN_SOURCE >= 700) \
    || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200809L)
    psiginfo(info, nullptr);
#else
    (void)info;
#endif
  }

 private:
  backward::details::handle<char*> _stack_content;
  bool _loaded;

#ifdef __GNUC__
  __attribute__((noreturn))
#endif
  static void
  sig_handler(int signo, siginfo_t* info, void* _ctx) {
    handleSignal(signo, info, _ctx);

    // try to forward the signal.
    raise(info->si_signo);

    // terminate the process immediately.
    puts("watf? exit");
    _exit(EXIT_FAILURE);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STACKTRACE_H_
