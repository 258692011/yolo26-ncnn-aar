cmake_minimum_required(VERSION 3.22.1)
project(yolo26 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 官方预编译动态库 ncnn-20260526-android-shared
# 实测(640 fp16, 第3次稳态): 动态 113ms < 静态 128ms -> 采用动态
# 与 AutoJS 的 lib/<abi>/ 部署模式一致: libncnn.so + libyolo26.so 同目录
add_library(ncnn SHARED IMPORTED)
set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libncnn.so)

find_library(JNIGRAPHICS_LIB jnigraphics)
find_library(ANDROID_LIB android)
find_library(LOG_LIB log)

add_library(yolo26 SHARED yolo26_jni.cpp)
target_include_directories(yolo26 PRIVATE ncnn_include/ncnn)
target_link_libraries(yolo26 ncnn ${JNIGRAPHICS_LIB} ${ANDROID_LIB} ${LOG_LIB})
