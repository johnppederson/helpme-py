add_library(helpme STATIC helpme.cc)
target_include_directories(helpme
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${EXTERNAL_INCLUDES}
)
target_link_libraries(helpme
    PRIVATE
        ${EXTERNAL_LIBRARIES}
)

if(HELPME_INSTALL_CPP)
    install(TARGETS helpme
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

pybind11_add_module(helpme_py MODULE pywrappers.cc)
target_link_libraries(helpme_py
    PRIVATE
        helpme
        ${EXTERNAL_LIBRARIES}
)
target_include_directories(helpme_py
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${EXTERNAL_INCLUDES}
)
install(TARGETS helpme_py LIBRARY DESTINATION .)
