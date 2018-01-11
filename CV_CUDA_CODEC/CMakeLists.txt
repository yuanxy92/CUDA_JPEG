# CMakeLists.txt for generic camera driver
# author: Shane Yuan
# date: Jan 18, 2018
#

set(subname "CUDACodec")

set(MY_HEADER_FILES
    Exceptions.h
    helper_string.h
    helper_cuda.h
    video_decoder.hpp
)

set(MY_SOURCE_FILES 
    main.cpp
    video_decoder.cpp
)

cuda_add_executable(${subname}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
)

target_link_libraries(${subname}
    ${OpenCV_LIBS}
    ${CUDA_npp_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_nppi_LIBRARY}
    cuda 
    nvcuvid
    cudart.lib
)
