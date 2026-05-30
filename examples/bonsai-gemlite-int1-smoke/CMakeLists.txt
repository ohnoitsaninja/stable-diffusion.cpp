set(TARGET sd-bonsai-gemlite-int1-smoke)

add_executable(${TARGET}
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
