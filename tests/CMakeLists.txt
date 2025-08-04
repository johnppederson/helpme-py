# Add unit test subdirectory.
add_subdirectory(unittests)

include(GNUInstallDirs
    CTest
)

# Add C example.
add_executable(RunCWrapper fullexample.c)
target_link_libraries(RunCWrapper PRIVATE helpme ${EXTERNAL_LIBRARIES})
set_target_properties(RunCWrapper PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(RunCWrapper PRIVATE ${PROJECT_SOURCE_DIR}/include ${EXTERNAL_INCLUDES})
add_test(NAME CWrapperTest COMMAND RunCWrapper)

# Add DHFR finite difference test.
set(NUM_FINDIF 30)
configure_file(data/dhfr_charges.txt . COPYONLY)
configure_file(data/dhfr_coords.txt . COPYONLY)
add_executable (DHFRFiniteDifference dhfr_findif.cpp)
target_link_libraries(DHFRFiniteDifference PRIVATE ${EXTERNAL_LIBRARIES})
target_include_directories(DHFRFiniteDifference PRIVATE ${PROJECT_SOURCE_DIR}/include ${EXTERNAL_INCLUDES})
add_test(NAME RunDHFRFiniteDifferenceTest COMMAND DHFRFiniteDifference 1 1 1 1 ${NUM_FINDIF})
if(MPI_FOUND)
    add_test(NAME RunDHFRFiniteDifferenceTestX
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 2 1 1 1 ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    add_test(NAME RunDHFRFiniteDifferenceTestY
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 2 1 1 ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    add_test(NAME RunDHFRFiniteDifferenceTestZ
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 1 2 1 ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    if(OpenMP_FOUND)
        add_test(NAME RunDHFRFiniteDifferenceTestXWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 2 1 1 ${TEST_NTHREADS} ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
        add_test(NAME RunDHFRFiniteDifferenceTestYWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 2 1 ${TEST_NTHREADS} ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
        add_test(NAME RunDHFRFiniteDifferenceTestZWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./DHFRFiniteDifference 1 1 2 ${TEST_NTHREADS} ${NUM_FINDIF} ${MPIEXEC_POSTFLAGS})
    endif()
else()
    if(OpenMP_FOUND)
        add_test(NAME RunDHFRFiniteDifferenceTestWithThreads
            COMMAND ./DHFRFiniteDifference 1 1 1 ${TEST_NTHREADS} ${NUM_FINDIF})
    endif()
endif()

# Add CXX example.
add_executable (RunCXXWrapper fullexample.cpp)
target_link_libraries(RunCXXWrapper PRIVATE ${EXTERNAL_LIBRARIES})
target_include_directories(RunCXXWrapper PRIVATE ${PROJECT_SOURCE_DIR}/include ${EXTERNAL_INCLUDES})
add_test(NAME CXXWrapperTest COMMAND RunCXXWrapper)
if(OpenMP_FOUND)
    add_test(NAME CXXWrapperTestWithThreads COMMAND RunCXXWrapper ${TEST_NTHREADS})
endif()

# Add C/CXX MPI example.
if(MPI_FOUND)
    add_executable (RunCXXMPIWrapper fullexample_parallel.cpp)
    target_link_libraries(RunCXXMPIWrapper PRIVATE helpme ${EXTERNAL_LIBRARIES})
    target_include_directories(RunCXXMPIWrapper PRIVATE ${PROJECT_SOURCE_DIR}/include ${EXTERNAL_INCLUDES})
    add_test(NAME CXXMPIWrapperTestX
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 2 1 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CXXMPIWrapperTestY
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 2 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CXXMPIWrapperTestZ
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 1 2 1 ${MPIEXEC_POSTFLAGS})
    if(OPENMP_FOUND)
        add_test(NAME CXXMPIWrapperTestXWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 2 1 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CXXMPIWrapperTestYWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 2 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CXXMPIWrapperTestZWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCXXMPIWrapper 1 1 2 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
    endif()

    add_executable (RunCMPIWrapper fullexample_parallel.c)
    target_link_libraries(RunCMPIWrapper PRIVATE helpme ${EXTERNAL_LIBRARIES})
    set_target_properties(RunCMPIWrapper PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(RunCMPIWrapper PRIVATE ${PROJECT_SOURCE_DIR}/include ${EXTERNAL_INCLUDES})
    add_test(NAME CMPIWrapperTestX
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 2 1 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CMPIWrapperTestY
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 2 1 1 ${MPIEXEC_POSTFLAGS})
    add_test(NAME CMPIWrapperTestZ
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 1 2 1 ${MPIEXEC_POSTFLAGS})
    if(OpenMP_FOUND)
        add_test(NAME CMPIWrapperTestXWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 2 1 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CMPIWrapperTestYWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 2 1 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
        add_test(NAME CMPIWrapperTestZWithThreads
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./RunCMPIWrapper 1 1 2 ${TEST_NTHREADS} ${MPIEXEC_POSTFLAGS})
    endif()
endif()

# Make sure examples are installed.
if(HELPME_INSTALL_CPP)
    set(EXAMPLEFILES fullexample.c fullexample.cpp fullexample.py)
    if(HAVE_MPI)
        list(APPEND EXAMPLEFILES fullexample_parallel.c fullexample_parallel.cpp)
    endif()
    install(FILES ${EXAMPLEFILES} DESTINATION examples)
endif()
