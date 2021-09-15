from conans import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.layout import cmake_layout


class OneflowConan(ConanFile):
    name = "oneflow"

    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    requires = [
        "pybind11/2.7.1",
        "zlib/1.2.11",
        "re2/20210601",
        "protobuf/3.17.1",
        "opencv/3.4.12",
        "lz4/1.9.3",
        "nlohmann_json/3.10.2",
        "half/2.2.0",
        "gtest/1.11.0",
        "grpc/1.39.1",
        "gflags/2.2.2",
        "flatbuffers/1.12.0",
        "eigen/3.3.9",
        "glog/0.5.0"
    ]

    generators = "cmake_find_package", "cmake_paths"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

        self.options["opencv"].with_jpeg = "libjpeg-turbo"
        self.options["jasper"].with_libjpeg = "libjpeg-turbo"
        self.options["libtiff"].jpeg = "libjpeg-turbo"

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["oneflow"]
