# oneflow
mkdir build  
cd build  

On Linux:  
cmake ..  
make -j  
  
On Windows:  
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Debug  
MSBuild /p:Configuration=Debug ALL_BUILD.vcxproj  

On Mac with Xcode  
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug  






