from conans import ConanFile, tools, AutoToolsBuildEnvironment
from conans.util.files import sha256sum


class HwlocConan(ConanFile):
    name = "hwloc"
    version = "2.4.1"
    license = "BSD license"
    url = "https://github.com/open-mpi/hwloc"
    description = "The Hardware Locality (hwloc) software project aims at easing the process of discovering hardware resources in parallel architectures."
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }
    generators = "cmake"

    def system_requirements(self):
        self.output.warn(
            'libudev and libpciaccess are required on your system')

    def source(self):
        tools.get(
            url=f"{self.url}/archive/hwloc-{self.version}.tar.gz",
            sha256="02bda2a88435f8cb365140c16eb19594627a1c8c0992fee3a8c8d241303abd3e",
            destination="sources", strip_root=True)

    def build(self):
        with tools.chdir("sources"):
            autotools = AutoToolsBuildEnvironment(self)

            with tools.environment_append(autotools.vars):
                self.run("./autogen.sh")

            def conf_val(val):
                return "yes" if val else "no"

            autotools.configure(args=[
                "--disable-libxml2",
                f"--enable-shared={conf_val(self.options.shared)}",
                f"--enable-static={conf_val(not self.options.shared)}",
            ])
            autotools.make()
            autotools.install()

    def package_info(self):
        self.cpp_info.libs = ["hwloc"]
        self.cpp_info.system_libs = ["udev", "pciaccess"]
