#      _________   _____________________  ____  ______
#     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
#    / /_  / /| | \__ \ / / / /   / / / / / / / __/
#   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
#  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
#
#  http://www.inf.ethz.ch/personal/markusp/teaching/
#  How to Write Fast Numerical Code 263-2300 - ETH Zurich
#  Copyright (C) 2017 Alen Stojanov (astojanov@inf.ethz.ch)
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see http://www.gnu.org/licenses/.

cmake_minimum_required(VERSION 3.0.2)
project(pcm)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -O2 -std=c++11")
set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_CURRENT_LIST_DIR}/../bin)
set(LIBRARY_OUTPUT_PATH     ${CMAKE_CURRENT_LIST_DIR}/../bin)


	set(PERF_FILES
		../include/perf.h
		perf/perf.cpp
	)

	# Main Intel Perfomance Counter Monitor API
	set(PCM_FILES
		pcm/cpucounters.h
		pcm/cpucounters.cpp
		pcm/msr.h
		pcm/msr.cpp
		pcm/client_bw.h
		pcm/client_bw.cpp
		pcm/pci.h
		pcm/pci.cpp
		pcm/utils.h
		pcm/utils.cpp
	)

	if (UNIX)
		if (APPLE)
			# Compile the Mac OS X MSR Accessor Driver
			include_directories(pcm/MacMSRDriver/)
			set(MAC_MSR
				pcm/MacMSRDriver/MSRAccessor.h
				pcm/MacMSRDriver/MSRAccessor.cpp
				pcm/MacMSRDriver/DriverInterface.h
				pcm/MacMSRDriver/DriverInterface.c
				pcm/MacMSRDriver/PCIDriverInterface.h
				pcm/MacMSRDriver/PCIDriverInterface.cpp
			)
			find_library(IOKIT IOKit)
			add_library(pcm ${PERF_FILES} ${PCM_FILES} ${MAC_MSR})
			target_link_libraries(pcm ${IOKIT})
		else(APPLE)
			add_library(pcm ${PERF_FILES} ${PCM_FILES})
			target_link_libraries(pcm rt pthread)
		endif(APPLE)
	endif (UNIX)

	if(WIN32)
		if (MSVC)
			include (GenerateExportHeader)
			include_directories(pcm)
			add_definitions(-DUNICODE -D_UNICODE)

			# Compile Windows MSR drivers
			set(WIN_MSR
				pcm/PCM_Win/windriver.h
				pcm/winring0/OlsApi.h
				pcm/winring0/OlsApiInit.h
				pcm/winring0/OlsApiInitDef.h
				pcm/winring0/OlsApiInitExt.h
				pcm/winring0/OlsDef.h
				pcm/winpmem/winpmem.h
				pcm/winpmem/winpmem.cpp
			)

			# Compile the driver accessor. While the driver accessor is
			# part of the WIN_MSR package, it is also acessing VS.NET API,
			# and as result needs special set of compiler flags.
			set(DRIVER_ACCESS_FILES pcm/PCM_Win/restrictDriverAccess.cpp)
			add_library(driverAccessLib ${DRIVER_ACCESS_FILES})
			GENERATE_EXPORT_HEADER(driverAccessLib
					BASE_NAME driverAccessLib
					EXPORT_MACRO_NAME driverAccessLib_EXPORT
					EXPORT_FILE_NAME driverAccessLib_Export.h
					STATIC_DEFINE driverAccessLib_BUILT_AS_STATIC
			)
			set_target_properties (driverAccessLib PROPERTIES VS_DOTNET_REFERENCES "System;System.IO;System.Security")
			target_compile_options(driverAccessLib PRIVATE /clr)
			target_compile_options(driverAccessLib PRIVATE /EHa)

			# Wrap the whole package into an extrnal library
			if (BUILD_PCM_DLL)
				add_library(pcm SHARED ${PERF_FILES} ${PCM_FILES} ${WIN_MSR})
			else (BUILD_PCM_DLL)
				add_library(pcm ${PERF_FILES} ${PCM_FILES} ${WIN_MSR})
			endif (BUILD_PCM_DLL)
			target_link_libraries(pcm driverAccessLib)

			# Copy the third-party precompiled dependencies of Intel
			# Performance Counter Monitor in the binary output path.
			#
			# 1. `winpmem` Rekall Memory Forensic Framework [Release 1.6.0 Gotthard]
			#     https://github.com/google/rekall/releases
			# 2. `winring` Real Temp 3.70
			#     https://www.techpowerup.com/realtemp/
			# 3.  elevate - Command-Line UAC Elevation Utility
			#     http://code.kliu.org/misc/elevate/
			#
			configure_file(${CMAKE_CURRENT_LIST_DIR}/drivers/win/winpmem_64.sys     ${LIBRARY_OUTPUT_PATH}/Release/winpmem_64.sys         COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/drivers/win/winpmem_86.sys     ${LIBRARY_OUTPUT_PATH}/Release/winpmem_86.sys         COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/drivers/win/WinRing0.dll       ${LIBRARY_OUTPUT_PATH}/Release/WinRing0.dll           COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/drivers/win/WinRing0.sys       ${LIBRARY_OUTPUT_PATH}/Release/WinRing0.sys           COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/drivers/win/WinRing0x64.dll    ${LIBRARY_OUTPUT_PATH}/Release/WinRing0x64.dll        COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/drivers/win/WinRing0x64.sys    ${LIBRARY_OUTPUT_PATH}/Release/WinRing0x64.sys        COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/elevate/bin.x86-32/elevate.exe ${LIBRARY_OUTPUT_PATH}/Release/bin.x86-32/elevate.exe COPYONLY)
			configure_file(${CMAKE_CURRENT_LIST_DIR}/elevate/bin.x86-64/elevate.exe ${LIBRARY_OUTPUT_PATH}/Release/bin.x86-64/elevate.exe COPYONLY)

			file(WRITE ${CMAKE_CURRENT_LIST_DIR}/../bin/IntelPCM.bat "\
				@echo off \
				reg Query \"HKLM\\Hardware\\Description\\System\\CentralProcessor\\0\" | find /i \"x86\" > NUL && set OS=32BIT || set OS=64BIT \n \
				setlocal \n \
				cd %~dp0\\Release\\ \n \
				if %OS%==32BIT bin.x86-32\\elevate IntelPCM \n \
				if %OS%==64BIT bin.x86-64\\elevate IntelPCM \n \
			")

		else (MSVC)
			message(FATAL_ERROR  "\n\
				Visual Studio is required to build Intel PCM\n \
				 - Download it and install it from:\n \
				   https://www.visualstudio.com/vs/community/\n \
			")
		endif(MSVC)
	endif(WIN32)


include (GenerateExportHeader)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
GENERATE_EXPORT_HEADER(pcm
    BASE_NAME pcm
    EXPORT_MACRO_NAME pcm_EXPORT
    EXPORT_FILE_NAME pcm_Export.h
    STATIC_DEFINE pcm_BUILT_AS_STATIC
)
