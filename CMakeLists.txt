cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

################  Options: Overview and Not Otherwise Mentioned  ###############
#
#  <<<  CMake build overview  >>>
#
#    >>> cmake -H. -Bobjdir -DCMAKE_INSTALL_PREFIX=/path/to/helpme/install/dir ...
#    ...
#    -- Generating done
#    -- Build files have been written to: /current/dir/objdir
#
#    >>> cd objdir && make -j`getconf _NPROCESSORS_ONLN`
#    >>> make install
#
#  <<<  Compilers and flags  >>>
#
#    - CMAKE_C_COMPILER "C compiler"
#    - CMAKE_C_FLAGS "Additional C flags"
#    - CMAKE_CXX_COMPILER "C++ compiler"
#    - CMAKE_CXX_FLAGS "Additional C++ flags"
#    - CMAKE_Fortran_COMPILER "Fortran compiler (required for some add-ons)"
#    - CMAKE_Fortran_FLAGS "Additional Fortran flags"
#
#  <<<  Detecting dependencies and add-ons  >>>
#
#    - PYTHON_EXECUTABLE "Python interpreter to use (e.g., /path/to/bin/python2.7)"
#    - SPHINX_ROOT "Root directory for Sphinx: 'bin/sphinx-build' (or similar) should be in this dir."
#

# Get version info.
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "${Cyan}Configuring build for helPME-py version ${GIT_VERSION}${ColourReset}")
else()
    set(GIT_VERSION "0.0.0-unknown")
    message(STATUS "${Red}Git not found, project version info unavailable${ColourReset}")
endif()

# Define project information.
project(helpme
    VERSION ${GIT_VERSION}
    DESCRIPTION "helPME-py: an efficient library for particle mesh Ewald"
    HOMEPAGE_URL "http://www.github.com/johnppederson/helpme-py"
    LANGUAGES C CXX
)
add_compile_definitions(VERSION="${GIT_VERSION}")

# Append path to custom cmake macros.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include cmake custom macros.
include(custom_color_messages)
include(optionsTools)
include(CTest)

# Set C++ standards.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define build options.
option_with_default(TEST_NTHREADS "Number of tests used in threaded \
                                   tests, if OpenMP is used" 2)
option_with_default(ERF_SCREEN_VAL "The value at which to consider the \
                                    error function equal to one (a lower \
                                    value makes the error function \
                                    evaluation faster but less \
                                    accurate).  Values <=5 will fail \
                                    CTest suite." 6)
option_with_default(HELPME_INSTALL_CPP "Install C++ headers/libs" ON)
option_with_print(BUILD_DOCS "Build documentation" ON)
option_with_print(ENABLE_OpenMP "Enables OpenMP parallelization" ON)
option_with_print(ENABLE_MPI "Enables MPI parallelization" ON)
option_with_print(ENABLE_BLAS "Enables BLAS usage" ON)
option_with_flags(ENABLE_ASAN "Enables address sanitizer" OFF
                  "-fsanitize=address" "-fno-omit-frame-pointer")
option_with_flags(ENABLE_MSAN "Enables address sanitizer" OFF
                  "-fsanitize=memory" "-fno-omit-frame-pointer")
option_with_flags(ENABLE_TSAN "Enables thread sanitizer" OFF
                  "-fsanitize=thread" "-fno-omit-frame-pointer -pie")
option_with_flags(ENABLE_UBSAN "Enables undefined behavior sanitizer" OFF
                  "-fsanitize=undefined" "-fno-omit-frame-pointer")

# Set C/C++ debug flags.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# The C++ library is linked explicitly because exception handling on macOS appears to be broken otherwise.
# !!! Check the above.
# Handle different C++ libraries.
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    set(cpplib c++)
else()
    set(cpplib stdc++)
endif()

# Find FFTW.
include(FindFFTW)
find_package(FFTW REQUIRED)
if(HAVE_FFTWF)
    add_definitions(-DHAVE_FFTWF=1)
endif()
if(HAVE_FFTWD)
    add_definitions(-DHAVE_FFTWD=1)
endif()
if(HAVE_FFTWL)
    add_definitions(-DHAVE_FFTWL=1)
endif()

# Set external includes.
set(EXTERNAL_INCLUDES
    ${FFTW_INCLUDES}
)

# Set external libraries.
set(EXTERNAL_LIBRARIES
    ${cpplib}
    ${FFTW_LIBRARIES}
)

# Find BLAS, if enabled.
if(ENABLE_BLAS)
    # Attempt to distinguish MKL from other BLAS routines.  This doesn't work as-is :(
    # !!! Test to see if the above is true.
    include(FindMKL)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        add_definitions(-DHAVE_MKL)
        set(EXTERNAL_INCLUDES
            ${EXTERNAL_INCLUDES}
            ${MKL_INCLUDE_DIR}
        )
        set(BLAS_LIBRARIES ${MKL_GNU} ${MKL_SEQUENTIAL} ${MKL_CORE})
        set(BLAS_FOUND TRUE)
    else()
        find_package(BLAS QUIET)
    endif()
    if(BLAS_FOUND)
        add_definitions(-DHAVE_BLAS)
        set(EXTERNAL_LIBRARIES
            ${EXTERNAL_LIBRARIES}
            ${BLAS_LIBRARIES}
        )
        message(STATUS "${Cyan}Found BLAS: ${BLAS_LIBRARIES}${ColourReset}")
    else()
        message(STATUS "${Red}BLAS not found, continuing without BLAS${ColourReset}")
    endif()
endif()

# Find OpenMP, if enabled.
if(ENABLE_OpenMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        add_definitions(-DHAVE_OpenMP)
        set(EXTERNAL_LIBRARIES
            ${EXTERNAL_LIBRARIES}
            OpenMP::OpenMP_CXX
        )
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "${Cyan}Found OpenMP: ${OpenMP_CXX_LIBRARIES}${ColourReset}")
        message(STATUS "${Cyan}OpenMP C FLAGS: ${OpenMP_C_FLAGS}${ColourReset}")
        message(STATUS "${Cyan}OpenMP CXX FLAGS: ${OpenMP_CXX_FLAGS}${ColourReset}")
    else()
        message(STATUS "${Red}OpenMP not found, continuing without OpenMP${ColourReset}")
    endif()
endif()

# Find MPI, if enabled.
if(ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        add_definitions(-DHAVE_MPI)
        set(EXTERNAL_LIBRARIES
            ${EXTERNAL_LIBRARIES}
            MPI::MPI_CXX
        )
        message(STATUS "${Cyan}Found MPI: ${MPI_INCLUDE_PATH}${ColourReset}")
    else()
        message(STATUS "${Red}MPI not found, continuing without MPI${ColourReset}")
    endif()
endif()

# Pass the error function screening threshold to the compiler.
add_compile_definitions(ERF_SCREEN_VAL=${ERF_SCREEN_VAL})

# Find Python.
set(PYBIND11_NEWPYTHON ON)
set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")   # -.
set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}") #  |-These are for the scikit-build system
set(Python_LIBRARY "${PYTHON_LIBRARY}")         # -'
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add subdirectories.
add_subdirectory(src)
if(BUILD_TESTING)
    find_package(Catch2 3 REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_package(Sphinx REQUIRED)
    find_package(LATEX)
    if(LATEX_FOUND)
        message(STATUS "${Cyan}Found LaTeX${ColourReset}")
    else()
        message(STATUS "${Red}LaTeX not found, continuing without LaTeX${ColourReset}")
    endif()
    add_subdirectory(docs)
endif()

# Install headers.
if(HELPME_INSTALL_CPP)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION .)
endif()
