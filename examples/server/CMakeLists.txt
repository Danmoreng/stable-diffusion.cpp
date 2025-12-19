set(TARGET sd-server)

add_executable(${TARGET} main.cpp)

# Define the path to the web UI's output directory (relative to this CMakeLists.txt)
set(WEBUI_PUBLIC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/public")

# Check if the frontend has been built. If not, stop and instruct the user.
if(NOT EXISTS "${WEBUI_PUBLIC_DIR}/index.html")
    message(FATAL_ERROR "Web UI not found. Please build the frontend first by running:\n  cd ${CMAKE_CURRENT_SOURCE_DIR}/webui\n  npm install\n  npm run build\n")
endif()

# Add a post-build command to copy the web UI files to the executable's directory
add_custom_command(
    TARGET ${TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${WEBUI_PUBLIC_DIR}" "$<TARGET_FILE_DIR:${TARGET}>/public"
    COMMENT "Copying web UI files to destination..."
)

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)