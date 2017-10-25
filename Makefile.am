bin_PROGRAMS = ptdemo runtests

AM_CXXFLAGS = -std=c++14 -O3
AM_CPPFLAGS = -I/usr/include/eigen3/

CXXFLAGS =

ptdemo_SOURCES = examples/ptdemo.cpp src/sampler.cpp src/mhsampler.cpp \
                 src/ptsampler.cpp src/linalg.cpp src/rand_helpers.cpp

runtests_SOURCES = tests/tests.cpp \
                   tests/pt_tests.cpp tests/mh_tests.cpp tests/stats_tests.cpp \
                   src/sampler.cpp src/mhsampler.cpp \
                   src/ptsampler.cpp src/linalg.cpp src/rand_helpers.cpp
