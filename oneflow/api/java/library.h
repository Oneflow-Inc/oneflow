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
#ifndef ONEFLOW_API_JAVA_LIBRARY_H_
#define ONEFLOW_API_JAVA_LIBRARY_H_

#include <jni.h>
#include "jni_md.h"

#ifdef __cplusplus
extern "C" {
#endif

// Detect Endian
JNIEXPORT jint JNICALL Java_org_oneflow_OneFlow_getEndian(JNIEnv* env, jobject obj);

// init
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_setIsMultiClient(JNIEnv* env, jobject obj,
                                                                 jboolean is_multi_client);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_initDefaultSession(JNIEnv* env, jobject obj);
JNIEXPORT jboolean JNICALL Java_org_oneflow_OneFlow_isEnvInited(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_initEnv(JNIEnv* env, jobject obj, jint ctrl_port);
JNIEXPORT jlong JNICALL Java_org_oneflow_OneFlow_currentMachineId(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_initScopeStack(JNIEnv* env, jobject obj);
JNIEXPORT jboolean JNICALL Java_org_oneflow_OneFlow_isSessionInited(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_initSession(JNIEnv* env, jobject obj,
                                                            jstring device_tag);

// compile
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_loadModel(JNIEnv* env, jobject obj, jobject option);

// launch
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_startLazyGlobalSession(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_loadCheckpoint(JNIEnv* env, jobject obj,
                                                               jobject path);
JNIEXPORT jstring JNICALL Java_org_oneflow_OneFlow_getPushJobNames(JNIEnv* env, jobject obj);
JNIEXPORT jstring JNICALL Java_org_oneflow_OneFlow_getPullJobNames(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_runSinglePushJob(JNIEnv* env, jobject obj,
                                                                 jobject data, jobject shape,
                                                                 jint dtype_code, jstring job_name,
                                                                 jstring op_name);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_runInferenceJob(JNIEnv* env, jobject obj,
                                                                jstring jstr);
JNIEXPORT jobject JNICALL Java_org_oneflow_OneFlow_runPullJob(JNIEnv* env, jobject obj,
                                                              jstring job_name, jstring op_name);

// clean
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_stopLazyGlobalSession(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_destroyLazyGlobalSession(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_destroyEnv(JNIEnv* env, jobject obj);
JNIEXPORT void JNICALL Java_org_oneflow_OneFlow_setShuttingDown(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_LIBRARY_H_
