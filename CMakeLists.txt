# CMakeLists.txt for generic camera driver
# author: Shane Yuan
# date: Dec 29, 2017
#

cmake_minimum_required (VERSION 3.1)
set(name "CUDA_JPEG")
project ( ${name} )

# c++ version
set (CMAKE_CXX_STANDARD 11)

# opencv 
if ( WIN32 )
    set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
    set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
    set(WIN_HEADER_PATH "${PROJECT_SOURCE_DIR}/winheadfile")
    include_directories(${WIN_HEADER_PATH})
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
set(NVIDIA_VIDEO_CODEC_SDK E:/libs/Video_Codec_SDK)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(CUDA )
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )

# nv codec 
set(NV_CODEC_DIRS "E:/libs/NV_CODEC/inc")
include_directories(${NV_CODEC_DIRS})

# add sub-directory
add_subdirectory(NPP_JPEG)
#add_subdirectory(NV_CODEC_JPEG)
add_subdirectory(NVPipe)