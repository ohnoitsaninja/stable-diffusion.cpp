set(TARGET sd-lens-text-encoder-smoke)

add_executable(${TARGET}
    main.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

if(SD_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(${TARGET} PRIVATE SD_LENS_TEXT_ENCODER_USE_CUBLAS=1)
        enable_language(CUDA)
        target_sources(${TARGET} PRIVATE mxfp4_dequant_cuda.cu)
        set_source_files_properties(mxfp4_dequant_cuda.cu PROPERTIES LANGUAGE CUDA)
        set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES ${SD_CUDA_ARCHITECTURES})
        if(TARGET CUDA::cublasLt)
            target_compile_definitions(${TARGET} PRIVATE SD_LENS_TEXT_ENCODER_USE_CUBLASLT=1)
            target_link_libraries(${TARGET} PRIVATE CUDA::cublasLt)
        endif()
        target_link_libraries(${TARGET} PRIVATE CUDA::cublas CUDA::cudart)
    endif()
endif()

set(NATIVE_TARGET sd-lens-text-encoder-native-smoke)

add_executable(${NATIVE_TARGET}
    native_api_main.cpp
    ../../src/lens_gptoss_text_encoder.cpp
)
install(TARGETS ${NATIVE_TARGET} RUNTIME)
target_include_directories(${NATIVE_TARGET} PRIVATE ../../src)
target_link_libraries(${NATIVE_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${NATIVE_TARGET} PUBLIC c_std_11 cxx_std_17)

if(SD_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(${NATIVE_TARGET} PRIVATE SD_LENS_TEXT_ENCODER_USE_CUBLAS=1)
        enable_language(CUDA)
        target_sources(${NATIVE_TARGET} PRIVATE mxfp4_dequant_cuda.cu)
        set_source_files_properties(mxfp4_dequant_cuda.cu PROPERTIES LANGUAGE CUDA)
        set_property(TARGET ${NATIVE_TARGET} PROPERTY CUDA_ARCHITECTURES ${SD_CUDA_ARCHITECTURES})
        if(TARGET CUDA::cublasLt)
            target_compile_definitions(${NATIVE_TARGET} PRIVATE SD_LENS_TEXT_ENCODER_USE_CUBLASLT=1)
            target_link_libraries(${NATIVE_TARGET} PRIVATE CUDA::cublasLt)
        endif()
        target_link_libraries(${NATIVE_TARGET} PRIVATE CUDA::cublas CUDA::cudart)
    endif()
endif()
