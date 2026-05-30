set(TARGET sd-lens-transformer-smoke)

set(RUNTIME_CORE_TARGET sd-lens-transformer-runtime-core)

add_library(${RUNTIME_CORE_TARGET} STATIC
    main.cpp
)
target_compile_definitions(${RUNTIME_CORE_TARGET} PRIVATE
    SD_LENS_TRANSFORMER_SMOKE_NO_MAIN=1
    SD_LENS_TRANSFORMER_SMOKE_RUNTIME=1
)
target_link_libraries(${RUNTIME_CORE_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${RUNTIME_CORE_TARGET} PUBLIC c_std_11 cxx_std_17)

add_executable(${TARGET}
    main.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
if(SD_CUDA)
    if(TARGET CUDA::cublasLt)
        target_compile_definitions(${TARGET} PRIVATE SD_LENS_TRANSFORMER_USE_CUBLASLT=1)
        target_link_libraries(${TARGET} PRIVATE CUDA::cublasLt)
    endif()
endif()

set(API_TARGET sd-lens-transformer-smoke-api)

add_library(${API_TARGET} SHARED
    main.cpp
)
target_compile_definitions(${API_TARGET} PRIVATE
    SD_LENS_TRANSFORMER_SMOKE_NO_MAIN=1
    SD_LENS_TRANSFORMER_SMOKE_EXPORTS=1
)
install(TARGETS ${API_TARGET} RUNTIME)
target_link_libraries(${API_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${API_TARGET} PUBLIC c_std_11 cxx_std_17)
