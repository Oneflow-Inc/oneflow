#ifndef ONEFLOW_API_JAVA_UTIL_JNI_UTIL_H_
#define ONEFLOW_API_JAVA_UTIL_JNI_UTIL_H_

#include <jni.h>
#include <string>

inline std::string ConvertToString(JNIEnv* env, jstring jstr) {
  const char* cstr = env->GetStringUTFChars(jstr, NULL);
  std::string sstr = cstr;
  env->ReleaseStringUTFChars(jstr, cstr);
  return sstr;
}

inline jstring ConvertToJString(JNIEnv* env, std::string str) {
  return env->NewStringUTF(str.c_str());
}

inline std::string Subreplace(std::string ori_str, std::string sub_str, std::string new_str) {
  std::string::size_type pos = 0;
  while ((pos = ori_str.find(sub_str)) != std::string::npos) {
    ori_str.replace(pos, sub_str.length(), new_str);
  }
  return ori_str;
}

// 1 for little endian, 0 for big endian
// take 32bit machine for example
// lower memory ----> higher memory
// little endian: 0x01 0x00 0x00 0x00
// big endian:    0x00 0x00 0x00 0x01
inline int Endian() {
  int x = 1;
  char *y = (char*) &x;
  return *y;
}

#endif  // ONEFLOW_API_JAVA_UTIL_JNI_UTIL_H_
