// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//

#pragma once

#ifndef _WIN32_WINNT  // Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT \
  0x0502  // Change this to the appropriate value to target other versions of
          // Windows.
#endif

#include <process.h>
#include <sal.h>
#include <stdio.h>
#include <tchar.h>
#include <new>

#include "ndstatus.h"
#include "ndsupport.h"

class Timer {
 private:
  LARGE_INTEGER m_Start;
  LARGE_INTEGER m_Split;
  LARGE_INTEGER m_End;
  LARGE_INTEGER m_Freq;

 public:
  Timer() {
    m_Start.QuadPart = 0;
    m_End.QuadPart = 0;
    m_Freq.QuadPart = 0;
    m_Split.QuadPart = 0;
    ::QueryPerformanceFrequency(&m_Freq);
  }

  void Start() { ::QueryPerformanceCounter(&m_Start); }

  void Split() { ::QueryPerformanceCounter(&m_Split); }

  void End() { ::QueryPerformanceCounter(&m_End); }

  double Report() const { return ElapsedMicrosec(m_Start, m_End); }

  double ReportPreSplit() const { return ElapsedMicrosec(m_Start, m_Split); }

  double ReportPostSplit() const { return ElapsedMicrosec(m_Split, m_End); }

  static LONGLONG Frequency() {
    LARGE_INTEGER m_Freq;
    ::QueryPerformanceFrequency(&m_Freq);
    return m_Freq.QuadPart;
  }

 private:
  double ElapsedMicrosec(_In_ LARGE_INTEGER start,
                         _In_ LARGE_INTEGER end) const {
    return (end.QuadPart - start.QuadPart) * 1000000.0
           / (double)m_Freq.QuadPart;
  }
};

class CpuMonitor {
 private:
  double m_StartCpu;
  double m_SplitCpu;
  double m_EndCpu;
  LONGLONG m_StartIdle;
  LONGLONG m_SplitIdle;
  LONGLONG m_EndIdle;

 public:
  CpuMonitor()
      : m_StartCpu(0),
        m_SplitCpu(0),
        m_EndCpu(0),
        m_StartIdle(0),
        m_SplitIdle(0),
        m_EndIdle(0) {}

  void Start() {
    LONGLONG kernelTime;
    LONGLONG userTime;
    GetSystemTimes(reinterpret_cast<FILETIME*>(&m_StartIdle),
                   reinterpret_cast<FILETIME*>(&kernelTime),
                   reinterpret_cast<FILETIME*>(&userTime));
    m_StartCpu = (double)(userTime + kernelTime);
  }

  void End() {
    LONGLONG kernelTime;
    LONGLONG userTime;
    GetSystemTimes(reinterpret_cast<FILETIME*>(&m_EndIdle),
                   reinterpret_cast<FILETIME*>(&kernelTime),
                   reinterpret_cast<FILETIME*>(&userTime));
    m_EndCpu = (double)(userTime + kernelTime);
  }

  void Split() {
    LONGLONG kernelTime;
    LONGLONG userTime;
    GetSystemTimes(reinterpret_cast<FILETIME*>(&m_SplitIdle),
                   reinterpret_cast<FILETIME*>(&kernelTime),
                   reinterpret_cast<FILETIME*>(&userTime));
    m_SplitCpu = (double)(userTime + kernelTime);
  }

  double Report() const {
    return GetCpuTime(m_StartIdle, m_EndIdle, m_StartCpu, m_EndCpu);
  }

  double ReportPreSplit() const {
    return GetCpuTime(m_StartIdle, m_SplitIdle, m_StartCpu, m_SplitCpu);
  }

  double ReportPostSplit() const {
    return GetCpuTime(m_SplitIdle, m_EndIdle, m_SplitCpu, m_EndCpu);
  }

  static DWORD CpuCount() {
    SYSTEM_INFO SystemInfo;
    GetSystemInfo(&SystemInfo);
    return SystemInfo.dwNumberOfProcessors;
  }

 private:
  static double GetCpuTime(_In_ LONGLONG startIdle, _In_ LONGLONG endIdle,
                           _In_ double startCpu, _In_ double endCpu) {
    return (100.0 - (((endIdle - startIdle) * 100) / (endCpu - startCpu)))
           * CpuCount();
  }
};
