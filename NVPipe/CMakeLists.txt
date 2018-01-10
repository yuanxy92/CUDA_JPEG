# CMakeLists.txt for generic camera driver
# author: Shane Yuan
# date: Jan 18, 2018
#

set(subname "NVPipe")

set(MY_HEADER_FILES
)

set(MY_SOURCE_FILES 
    # main function
    main.cpp
)

set(NVPIPE_DIRS "E:/libs/NVPipe/include")
set(NVPIPE_LIBS_DEBUG "E:/libs/NVPipe/lib/nvpiped.lib")
set(NVPIPE_LIBS_RELEASE "E:/libs/NVPipe/lib/nvpipe.lib")
include_directories(${NVPIPE_DIRS})

cuda_add_executable(${subname}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
)

target_link_libraries(${subname}
    ${OpenCV_LIBS}
    debug ${NVPIPE_LIBS_DEBUG}  
    optimized ${NVPIPE_LIBS_RELEASE}
)
