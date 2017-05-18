# Oneflow

## 1. Build

For all platform, do

```txt
mkdir build && cd build
```

### 1.1 Linux 

```
cmake -DDOWNLOAD_THIRD_PARTY=ON ..  
make
cmake -DDOWNLOAD_THIRD_PARTY=OFF ..
make
```

### 1.2 Windows

if first cmake:
```
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Debug -DDOWNLOAD_THIRD_PARTY=ON
MSBuild /p:Configuration=Debug ALL_BUILD.vcxproj  
```
else:
```
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Debug -DDOWNLOAD_THIRD_PARTY=OFF
MSBuild /p:Configuration=Debug ALL_BUILD.vcxproj  
```

### 1.3 Mac with Xcode

```
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug 
```

## 2. Compile GoogleNet

### 2.1 Linux

```
cd build && make
../run.sh
```
