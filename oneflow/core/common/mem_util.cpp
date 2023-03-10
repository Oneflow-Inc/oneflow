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
#include "oneflow/core/common/mem_util.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"

#include <unistd.h>
#include <sys/sysinfo.h>

namespace oneflow {

namespace {
struct ProcStat {
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize = 0;
  long rss = 0;
};

Maybe<void> CPUSynchronize() {
  if (Singleton<VirtualMachine>::Get() != nullptr) { return vm::CurrentRankSync(); }
  return Maybe<void>::Ok();
}

}  // namespace

// Reference: https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
void ProcessMemUsage(double* vm_usage, double* resident_set) {
  *vm_usage = 0.0;
  *resident_set = 0.0;

#ifdef __linux__
  // 'file' stat seems to give the most reliable results
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
  ProcStat proc_stat;
  stat_stream >> proc_stat.pid >> proc_stat.comm >> proc_stat.state >> proc_stat.ppid
      >> proc_stat.pgrp >> proc_stat.session >> proc_stat.tty_nr >> proc_stat.tpgid
      >> proc_stat.flags >> proc_stat.minflt >> proc_stat.cminflt >> proc_stat.majflt
      >> proc_stat.cmajflt >> proc_stat.utime >> proc_stat.stime >> proc_stat.cutime
      >> proc_stat.cstime >> proc_stat.priority >> proc_stat.nice >> proc_stat.num_threads
      >> proc_stat.itrealvalue >> proc_stat.starttime >> proc_stat.vsize
      >> proc_stat.rss;  // don't care about the rest

  stat_stream.close();

  long page_size_kb = sysconf(_SC_PAGE_SIZE);  // in case x86-64 is configured to use 2MB pages
  // return with MB
  *vm_usage = proc_stat.vsize >> 20;
  // return with MB
  *resident_set = (proc_stat.rss * page_size_kb) >> 20;
#endif  // __linux__
}

Maybe<double> GetCPUMemoryUsed() {
  JUST(CPUSynchronize());
  double vm_ = 0, rss_ = 0;
  ProcessMemUsage(&vm_, &rss_);
  return rss_;
}

std::string FormatMemSize(uint64_t size) {
  std::ostringstream os;
  os.precision(1);
  os << std::fixed;
  if (size <= 1024UL) {
    os << size << " Bytes";
  } else if (size <= 1048576UL) {
    os << ((float)size / 1024.0) << " KB";
  } else if (size <= 1073741824UL) {
    os << ((float)size / 1048576.0) << " MB";
  } else {
    os << ((float)size / 1073741824.0) << " GB";
  }
  return os.str();
}

}  // namespace oneflow
