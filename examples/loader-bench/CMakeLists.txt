set(TARGET sd-loader-bench)

add_executable(${TARGET}
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
