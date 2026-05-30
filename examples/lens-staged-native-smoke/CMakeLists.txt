set(TARGET sd-lens-staged-native-smoke)

add_executable(${TARGET}
    ../common/log.cpp
    ../common/media_io.cpp
    main.cpp
    ../../src/lens_gptoss_text_encoder.cpp
    ../../src/lens_gptoss_tokenizer.cpp
    ../../src/lens_staged_pipeline.cpp
    ../../src/lens_transformer_runtime.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ../../src ../common)
target_link_libraries(${TARGET} PRIVATE stable-diffusion sd-lens-transformer-runtime-core ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE SD_LENS_STAGED_PIPELINE_ENABLE_IMPL=1)

if(SD_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(${TARGET} PRIVATE SD_LENS_TEXT_ENCODER_USE_CUBLAS=1)
        enable_language(CUDA)
        target_sources(${TARGET} PRIVATE ../lens-text-encoder-smoke/mxfp4_dequant_cuda.cu)
        set_source_files_properties(../lens-text-encoder-smoke/mxfp4_dequant_cuda.cu PROPERTIES LANGUAGE CUDA)
        set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES ${SD_CUDA_ARCHITECTURES})
        if(TARGET CUDA::cublasLt)
            target_compile_definitions(${TARGET} PRIVATE SD_LENS_TEXT_ENCODER_USE_CUBLASLT=1)
            target_link_libraries(${TARGET} PRIVATE CUDA::cublasLt)
        endif()
        target_link_libraries(${TARGET} PRIVATE CUDA::cublas CUDA::cudart)
    endif()
endif()
