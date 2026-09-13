set(TARGET llama-deepseek-v41-trace)
add_executable(${TARGET} llama-trace.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
