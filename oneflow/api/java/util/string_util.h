#ifndef ONEFLOW_API_JAVA_UTIL_STRING_UTIL_H_
#define ONEFLOW_API_JAVA_UTIL_STRING_UTIL_H_

#include <jni.h>
#include <string>

inline std::string convert_jstring_to_string(JNIEnv* env, jstring jstr) {
  const char* cstr = env->GetStringUTFChars(jstr, NULL);
  std::string sstr = cstr;
  env->ReleaseStringUTFChars(jstr, cstr);
  return sstr;
}

inline jstring convert_string_to_jstring(JNIEnv* env, std::string str) {
  return env->NewStringUTF(str.c_str());
}

#endif  // ONEFLOW_API_JAVA_UTIL_STRING_UTIL_H_
