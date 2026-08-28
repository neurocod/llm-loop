cmake_minimum_required(VERSION 3.21)

# CMP0091: the MSVC runtime is picked by CMAKE_MSVC_RUNTIME_LIBRARY below, not
# by a flag baked into CMAKE_CXX_FLAGS_<CONFIG>.
cmake_policy(SET CMP0091 NEW)

project(ask_user_gate LANGUAGES CXX)

add_executable(ask_user_gate ask_user_gate.cpp)
target_compile_features(ask_user_gate PRIVATE cxx_std_23)

# The binary must land next to hooks.json: `HERE` is its own directory, and both
# `../bin/<tool>` (the paths the refusals name) and the wiring self-test resolve
# from there. Every per-config spelling is set because a multi-config generator
# would otherwise bury it in Release/ and break both.
set(GATE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hooks")
set_target_properties(ask_user_gate PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${GATE_OUTPUT_DIR}"
	RUNTIME_OUTPUT_DIRECTORY_DEBUG "${GATE_OUTPUT_DIR}"
	RUNTIME_OUTPUT_DIRECTORY_RELEASE "${GATE_OUTPUT_DIR}"
	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${GATE_OUTPUT_DIR}"
	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${GATE_OUTPUT_DIR}")

# Size, not throughput, is what this binary's optimiser settings buy. It runs
# for a few milliseconds on a ~100-character string, so nothing it computes is
# worth a flag; what it does ~100k times a month is START, and a start costs an
# image to map and its pages to fault in. Every setting below therefore shrinks
# the image or drops code that is never reached, and cpp/build.py prints the
# resulting size on every build so a regression is visible, not argued about.
if(MSVC)
	# CMake's own Release flags carry /O2 (speed) and its base flags carry /GR
	# (RTTI); cl warns D9025 when the opposite is appended instead of replacing
	# them, so they are replaced.
	string(REGEX REPLACE "/GR( |$)" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	string(REGEX REPLACE "/O2( |$)" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
	string(REGEX REPLACE "/Ob2( |$)" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

	# /utf-8: the raw string literals are the refusal text, and the source is
	# UTF-8 whatever the machine's ANSI codepage says.
	target_compile_options(ask_user_gate PRIVATE /utf-8 /W4 /permissive-)
	# std::getenv is only ever read here, never kept across a setenv.
	target_compile_definitions(ask_user_gate PRIVATE _CRT_SECURE_NO_WARNINGS)

	#   /O1        optimise for size (implies /Os /Oy /Ob2 /GF /Gy -- and /Gy,
	#              function-level COMDATs, is what gives /OPT:REF something to
	#              drop in the first place)
	#   /GR-       no RTTI: nothing here uses dynamic_cast or typeid
	#   /Gw        per-variable COMDATs, so /OPT:REF reaches unused DATA too
	#   /OPT:REF   drop every COMDAT nothing reaches -- the "unused symbols" half
	#   /OPT:ICF   fold the ones that came out identical
	#   /DEBUG:NONE keeps a PDB out of hooks/, a shipped directory, not a build one
	# Written into the per-config flag variables rather than as generator
	# expressions: a genex must fit in ONE argument, and /Zc: and /OPT: options
	# carry colons of their own, which is a fight not worth having here.
	set(GATE_SIZE_COMPILE "/O1 /GR- /Gw")
	set(GATE_SIZE_LINK "/OPT:REF /OPT:ICF /INCREMENTAL:NO /DEBUG:NONE")
	foreach(config RELEASE MINSIZEREL)
		set(CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}} ${GATE_SIZE_COMPILE}")
		set(CMAKE_EXE_LINKER_FLAGS_${config} "${CMAKE_EXE_LINKER_FLAGS_${config}} ${GATE_SIZE_LINK}")
	endforeach()
	# Static CRT. This binary is started ~100k times a month, so the two DLL
	# loads it saves are the point; it also makes the built gate self-contained,
	# which matters for a file that ships inside a plugin.
	set_property(TARGET ask_user_gate PROPERTY
		MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	# Whole-program optimisation, set through the property rather than a bare
	# /GL so the generator also puts /LTCG on the link step.
	set_property(TARGET ask_user_gate PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
	set_property(TARGET ask_user_gate PROPERTY INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
else()
	target_compile_options(ask_user_gate PRIVATE -Wall -Wextra)
	foreach(config RELEASE MINSIZEREL)
		set(CMAKE_CXX_FLAGS_${config}
			"${CMAKE_CXX_FLAGS_${config}} -Os -fno-rtti -ffunction-sections -fdata-sections")
		set(CMAKE_EXE_LINKER_FLAGS_${config}
			"${CMAKE_EXE_LINKER_FLAGS_${config}} -Wl,--gc-sections -s")
	endforeach()
endif()
