# CMakeLists.txt for generic camera driver
# author: Shane Yuan
# date: Jan 18, 2018
#

set(subname "NV_CODEC_JPEG")

add_definitions(-DINIT_CUDA_D3D11)

set(MY_HEADER_FILES
    cudaProcessFrame.h
    FrameQueue.h
    ImageDX.h
    VideoDecoder.h
    VideoParser.h
    VideoSource.h
)

set(MY_SOURCE_FILES 
    cudaProcessFrame.cpp
    FrameQueue.cpp
    ImageDX.cpp
    main.cpp
    VideoDecoder.cpp
    VideoParser.cpp
    VideoSource.cpp
)

set(NV_CODEC_LIBS
    "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/d3d11.lib"
    "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/dxgi.lib" 
    "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/AdvAPI32.Lib"
    "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/odbc32.Lib"
    "C:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/odbccp32.Lib"
)

cuda_add_executable(${subname}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
)

target_link_libraries(${subname}
    ${NV_CODEC_LIBS}
    cuda 
    nvcuvid
    cudart.lib
    "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/nvToolsExt64_1.lib"
)
