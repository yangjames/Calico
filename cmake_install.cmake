# Install script for directory: /home/jpollak/src/jbcpollak/Calico

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libaccelerometer.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libaccelerometer_cost_functor.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libaccelerometer_models.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libapriltags.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libbatch_optimizer.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libcamera.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libcamera_cost_functor.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libcamera_models.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libgyroscope.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libgyroscope_cost_functor.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libgyroscope_models.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libtrajectory.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libworld_model.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/calico" TYPE STATIC_LIBRARY FILES "/home/jpollak/src/jbcpollak/Calico/build/lib/libaprilgrid_detector.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/calico/CalicoConfig.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/calico/CalicoConfig.cmake"
         "/home/jpollak/src/jbcpollak/Calico/CMakeFiles/Export/lib/cmake/calico/CalicoConfig.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/calico/CalicoConfig-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/calico/CalicoConfig.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/calico" TYPE FILE FILES "/home/jpollak/src/jbcpollak/Calico/CMakeFiles/Export/lib/cmake/calico/CalicoConfig.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/calico" TYPE FILE FILES "/home/jpollak/src/jbcpollak/Calico/CMakeFiles/Export/lib/cmake/calico/CalicoConfig-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/calico" TYPE FILE FILES
    "/home/jpollak/src/jbcpollak/Calico/calico/profiler.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/trajectory.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/bspline.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/test_utils.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/optimization_utils.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/batch_optimizer.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/typedefs.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/matchers.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/status_builder.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/geometry.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/world_model.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/statusor_macros.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/calico/sensors" TYPE FILE FILES
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/gyroscope_models.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/camera.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/accelerometer_cost_functor.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/sensor_base.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/gyroscope_cost_functor.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/camera_cost_functor.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/camera_models.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/accelerometer_models.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/gyroscope.h"
    "/home/jpollak/src/jbcpollak/Calico/calico/sensors/accelerometer.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jpollak/src/jbcpollak/Calico/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
