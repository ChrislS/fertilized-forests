configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fertilized/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/fertilized/version.h @ONLY)

# include target for headers as workaround to havae them shown up in a IDE
FILE(GLOB_RECURSE LibFiles "*.h")
add_custom_target(headers SOURCES ${LibFiles})

install (DIRECTORY fertilized DESTINATION include
  PATTERN "*.in" EXCLUDE)
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/fertilized/version.h
  DESTINATION include/fertilized/)
