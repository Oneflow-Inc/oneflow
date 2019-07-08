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


tf_cc_shared_object(
    name = "libxla_computation_client.so",
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-s",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:interpreter_plugin",
#        "//tensorflow/compiler/jit:xla_cpu_jit",
#        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)
