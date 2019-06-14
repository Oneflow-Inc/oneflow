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
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:svd",
        "//tensorflow/compiler/xla/rpc:grpc_stub",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "test",
    srcs = ["test.cc"],
    deps = [
        "//tensorflow/compiler/lib:libxla_computation_client.so",
    ],
)
