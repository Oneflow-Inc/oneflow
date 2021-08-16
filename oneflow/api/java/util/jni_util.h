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

inline jobject GetOptionField(JNIEnv* env, jobject obj, const char* field_name, 
                              const char* signature) {
  jclass option_class = (*env).FindClass("org/oneflow/Option");
  jfieldID fid = (*env).GetFieldID(option_class, field_name, signature);
  jobject field_obj = (*env).GetObjectField(obj, fid);
  return field_obj;
}

inline int GetIntFromField(JNIEnv* env, jobject obj) {
  jclass integer_class = (*env).FindClass("java/lang/Integer");
  jmethodID mid = (*env).GetMethodID(integer_class, "intValue", "()I");
  int value = (*env).CallIntMethod(integer_class, mid);
  return value;
}

inline void SetStringField(JNIEnv* env, jobject obj, const char* field_name,
                           const char* signature, jstring value) {
  jclass option_class = (*env).FindClass("org/oneflow/Option");
  jfieldID fid = (*env).GetFieldID(option_class, field_name, signature);
  (*env).SetObjectField(obj, fid, (jobject) value);
}

#endif  // ONEFLOW_API_JAVA_UTIL_JNI_UTIL_H_
