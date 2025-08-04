# Add any new tests to this list or the one below!
set(SERIAL_TEST_SOURCES
    unittest-cartesiantransform.cpp
    unittest-cdq-field-findif.cpp
    unittest-coulombkappasweep.cpp
    unittest-d-field-kappasweep.cpp
    unittest-dipolereference.cpp
    unittest-dispersionkappasweep.cpp
    unittest-fft.cpp
    unittest-fletcher.cpp
    unittest-fullrun.cpp
    unittest-fullrun-compressed.cpp
    unittest-fullrun-isodipole-virial.cpp
    unittest-fullrun-multipoles.cpp
    unittest-gammafunction.cpp
    unittest-gridsize.cpp
    unittest-individualshells.cpp
    unittest-lattice.cpp
    unittest-latticeupdates.cpp
    unittest-matrix.cpp
    unittest-minimumimage.cpp
    unittest-powers.cpp
    unittest-potential.cpp
    unittest-splines.cpp
    unittest-string.cpp
    unittest-tensor.cpp
)

if(HAVE_MPI)
    set(PARALLEL_TEST_SOURCES
        unittest-coulomb-rec-parallel.cpp
        unittest-dispersion-rec-parallel.cpp
    )
endif()

foreach(test_file IN LISTS SERIAL_TEST_SOURCES)
    string(REPLACE ".cpp" "" test ${test_file})
    add_executable(${test} ${test_file})
    target_link_libraries(${test} PRIVATE ${EXTERNAL_LIBRARIES} Catch2::Catch2WithMain)
    target_compile_definitions(${test} PRIVATE -DHELPME_TESTS_NTHREADS=${TEST_NTHREADS})
    target_include_directories(${test} PRIVATE ${CATCH_HEADER_DIR} ${PROJECT_SOURCE_DIR}/include)
    set_property(TARGET ${test} PROPERTY CXX_STANDARD 17)
    add_test(NAME ${test}
        COMMAND ${test}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unittests
    )
endforeach()

foreach(test_file IN LISTS PARALLEL_TEST_SOURCES)
    string(REPLACE ".cpp" "" test ${test_file})
    add_executable(${test} ${test_file})
    target_link_libraries(${test} PRIVATE ${EXTERNAL_LIBRARIES} ${MPI_CXX_LIBRARIES} Catch2::Catch2WithMain)
    target_compile_definitions(${test} PRIVATE -DHELPME_TESTS_NTHREADS=${TEST_NTHREADS} -DHAVE_MPI=1)
    target_include_directories(${test} PRIVATE ${CATCH_HEADER_DIR} ${MPI_CXX_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include)
    set_property(TARGET ${test} PROPERTY CXX_STANDARD 17)
    add_test(NAME ${test}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./${test} ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unittests
    )
endforeach()

foreach(test_file IN LISTS SERIAL_TEST_SOURCES PARALLEL_TEST_SOURCES)
    string(REPLACE ".cpp" "" test ${test_file})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GCC")
        target_compile_options(${test}  PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors -Wmissing-noreturn -Wno-int-in-bool-context)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
        target_compile_options(${test}  PRIVATE -Wweak-vtables -Wexit-time-destructors -Wglobal-constructors)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC" )
        target_compile_options(${test}  PRIVATE /W4 /w44265 /WX)
    endif()
endforeach()
