# Define input and output directories.
set(DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doxygen")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(SPHINX_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sphinx")
set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

# Configure Doxygen.
file(MAKE_DIRECTORY "${DOXYGEN_OUTPUT_DIR}")
configure_file(
    "${DOXYGEN_INPUT_DIR}/Doxyfile.in"
    "${DOXYGEN_OUTPUT_DIR}/Doxyfile"
    @ONLY
)

# Add command to generate Doxygen XML file.
add_custom_command(
    OUTPUT "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
    COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYGEN_OUTPUT_DIR}/Doxyfile"
    WORKING_DIRECTORY "${DOXYGEN_OUTPUT_DIR}"
    DEPENDS "${DOXYGEN_OUTPUT_DIR}/Doxyfile"
    COMMENT "Generating Doxygen XML docs..."
    VERBATIM
)
add_custom_target(DoxygenDocs
    DEPENDS "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
)

# Add Sphinx sources.
set(SOURCES_SPHINX
    bibliography.rst
    index.rst
    introduction.rst
)
set(SPHINX_INPUT_SOURCES)
foreach(SOURCE_FILE ${SOURCES_SPHINX})
    list(APPEND
        SPHINX_INPUT_SOURCES
        "${SPHINX_INPUT_DIR}/${SOURCE_FILE}"
    )
endforeach()

# Configure Sphinx.
file(MAKE_DIRECTORY "${SPHINX_OUTPUT_DIR}/html")
configure_file(
    "${SPHINX_INPUT_DIR}/conf.py.in"
    "${SPHINX_OUTPUT_DIR}/conf.py"
    @ONLY
)

# Add command to generate Sphinx HTML file.
add_custom_command(
    OUTPUT "${SPHINX_OUTPUT_DIR}/html/index.html"
    COMMAND ${SPHINX_EXECUTABLE} -q -b html
            -c "${SPHINX_OUTPUT_DIR}" "${SPHINX_INPUT_DIR}"
            "${SPHINX_OUTPUT_DIR}/html"
    DEPENDS ${SPHINX_INPUT_SOURCES} "${SPHINX_OUTPUT_DIR}/conf.py"
            "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
    COMMENT "Generating Sphinx HTML docs..."
    VERBATIM
)
add_custom_target(SphinxHtml
    DEPENDS "${SPHINX_OUTPUT_DIR}/html/index.html"
)

# Configure optional LaTeX/PDF.
file(MAKE_DIRECTORY "${SPHINX_OUTPUT_DIR}/latex")

# Add command to generate optional LaTeX/PDF.
add_custom_command(
    OUTPUT "${SPHINX_OUTPUT_DIR}/latex/Makefile"
    COMMAND ${SPHINX_EXECUTABLE} -q -b latex
            -c "${SPHINX_OUTPUT_DIR}" "${SPHINX_INPUT_DIR}"
            "${SPHINX_OUTPUT_DIR}/latex/Makefile"
    DEPENDS ${SPHINX_INPUT_SOURCES}
            "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
    COMMENT "Generating Sphinx LaTeX sources..."
    VERBATIM
)
add_custom_target(SphinxLatex
    DEPENDS "${SPHINX_OUTPUT_DIR}/latex/Makefile"
)

#if(LATEX_FOUND)
#    add_custom_command(
#        OUTPUT "${SPHINX_OUTPUT_DIR}/latex/helPME.pdf"
#        COMMAND make
#        WORKING_DIRECTORY "${SPHINX_OUTPUT_DIR}/latex"
#        DEPENDS "${SPHINX_OUTPUT_DIR}/latex/Makefile"
#        COMMENT "Generating LaTeX PDF..."
#        VERBATIM
#    )
#    add_custom_target(SphinxPdf
#        DEPENDS "${SPHINX_OUTPUT_DIR}/latex/helPME.pdf"
#    )
#endif()

# Add unified build target.
add_custom_target(docs ALL
    DEPENDS SphinxHtml
#    $<$<BOOL:${LATEX_FOUND}>:SphinxPdf>
)

# Install documentation.
install(
    DIRECTORY "${SPHINX_OUTPUT_DIR}/html"
    DESTINATION share/doc
)
#if(LATEX_FOUND)
#    install(
#        DIRECTORY "${SPHINX_OUTPUT_DIR}/latex/helPME.pdf"
#        DESTINATION share/doc
#    )
#endif()
