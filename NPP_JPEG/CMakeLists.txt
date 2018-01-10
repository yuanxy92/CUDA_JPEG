# CMakeLists.txt for generic camera driver
# author: Shane Yuan
# date: Jan 18, 2018
#

set(subname "NPP_JPEG")

set(MY_HEADER_FILES
    Exceptions.h
    helper_cuda.h
    helper_string.h
    NPPJpegCoder.h
)

set(MY_SOURCE_FILES 
    # C++ source file
    NPPJpegCoder.cpp
    # main function
    main.cpp
)

cuda_add_executable(${subname}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
)

target_link_libraries(${subname}
    ${OpenCV_LIBS}
    ${CUDA_npp_LIBRARY}
    ${CUDA_nppc_LIBRARY}
)

