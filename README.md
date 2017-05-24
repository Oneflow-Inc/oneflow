# Oneflow

### 1.1 Linux 

```
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```

### 1.2 Windows

```
./build.bat third_party
./build.bat
```

### 1.3 Mac with Xcode

```
./build_xcode.sh third_party
./build_xcode.sh
```

## 2. Compile GoogleNet

### 2.1 Linux

```
cd build && make
../run.sh
```
