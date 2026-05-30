set(TARGET sd-bonsai-gemlite-int1-kernel-bench)

add_executable(${TARGET}
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
