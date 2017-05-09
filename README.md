# oneflow
mkdir build  
cd build  

On Linux:  
Please run the following command if you for the first time set up the dev environment  
cmake .. -Doneflow_DOWNLOAD_THIRD_PARTY=ON  

Just run  
cmake ..  
if you already download and build all the third party codes  

make -j  
  
On Windows:  
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Debug -Doneflow_DOWNLOAD_THIRD_PARTY=ON  
or run  
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Debug  
if you already download and build all the third party codes  

Build all the projects:  
MSBuild /p:Configuration=Debug ALL_BUILD.vcxproj  

On Mac with Xcode: 
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug -Doneflow_DOWNLOAD_THIRD_PARTY=ON  
or  
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug  
if you already download and build all the third party codes  