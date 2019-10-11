licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_binary",
    "tf_cc_shared_object",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

cc_library(
    name = "xla_runtime_util",
    srcs = [
        "xla_runtime_util.cc",
        "xla_runtime_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/stream_executor:stream",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service/cpu:cpu_executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
    ],
)

tf_cc_shared_object(
    name = "libxla_core.so",
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z muldefs",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":xla_runtime_util",
        "//tensorflow/compiler/jit:xla_tensor",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:cpu_plugin",
#        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla/lib:util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ] + if_cuda([
        "//tensorflow/compiler/xla/service:gpu_plugin",
#        "//tensorflow/compiler/jit:xla_gpu_jit",
    ]),
)
