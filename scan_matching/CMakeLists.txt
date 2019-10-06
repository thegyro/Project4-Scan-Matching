set(SOURCE_FILES
    "scan_matching.cpp"
    )

cuda_add_library(scan_matching
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
