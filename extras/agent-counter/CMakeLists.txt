# ---------------------------------------------------------------------------
# AgentCounter — on-screen badge counting the background agents that are running
# (target: AgentCounter). Started on demand by `src/llm_agents/agent_counter.py`.
#
# Deliberately plain Qt (Core/Gui/Network/Widgets) with no pch and no other
# library: it must build and start in the time between "an agent begins" and
# "the operator glances at the screen".
#
# Builds two ways, with the same target:
#   * on its own: `cmake -S extras/agent-counter -B build -DCMAKE_PREFIX_PATH=<Qt6>`;
#   * from a host project's tree via `add_subdirectory`, which may already have
#     found Qt and run `qt_standard_project_setup()`.
# Either way a missing Qt6 is a skip with a message, never a configure error:
# the Python bridges work without the badge, so nothing may require it.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)

# Not PROJECT_IS_TOP_LEVEL: before this file's own project() it still holds
# the PARENT project's value, which is TRUE for a host's root project.
set(_agent_counter_standalone OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_agent_counter_standalone ON)
    project(AgentCounter LANGUAGES CXX)
endif()

find_package(Qt6 6.5 QUIET COMPONENTS Core Gui Network Widgets)
if(NOT Qt6_FOUND OR NOT TARGET Qt6::Widgets OR NOT TARGET Qt6::Network)
    message(STATUS "AgentCounter: Qt6 Core/Gui/Network/Widgets not found, skipped")
    return()
endif()

if(_agent_counter_standalone)
    qt_standard_project_setup()
endif()

qt_add_executable(AgentCounter WIN32
    main.cpp
    Badge.cpp
    BadgeTest.cpp
    Registry.cpp
    SelfTest.cpp
    # Headers listed as sources so AUTOMOC discovers Q_OBJECT classes.
    Badge.h
    BadgeTest.h
    Registry.h
    SelfTest.h
)

target_link_libraries(AgentCounter PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Network
    Qt6::Widgets
)

target_compile_features(AgentCounter PRIVATE cxx_std_23)

# Provider icons are a BUILD INPUT, not part of this project: the providers'
# logos are their owners' marks and do not ship under this repository's
# licence. A directory holding `claude.png` and `codex.png` is embedded as
# `:/agents/<provider>.png`; without one (or with an unusable one) the badge
# paints its own neutral marks (Badge.cpp, neutralIcon) — a warning, never a
# configure error, per the promise at the top of this file.
set(AGENT_COUNTER_ICON_DIR "" CACHE PATH
    "Directory with claude.png and codex.png to embed as provider icons; empty = neutral built-in marks")
set(_agent_counter_icons "")
if(AGENT_COUNTER_ICON_DIR)
    foreach(provider IN ITEMS claude codex)
        set(_icon "${AGENT_COUNTER_ICON_DIR}/${provider}.png")
        set(_icon_header "")
        if(EXISTS "${_icon}")
            file(READ "${_icon}" _icon_header LIMIT 8 HEX)
        endif()
        if(_icon_header STREQUAL "89504e470d0a1a0a")
            list(APPEND _agent_counter_icons "${_icon}")
        else()
            # A host may keep its *.png in Git LFS; an unfetched pointer would
            # otherwise surface as a corrupt resource at runtime.
            message(WARNING "AgentCounter: ${_icon} is missing or not a PNG (a Git LFS pointer? run git lfs pull); "
                            "using the neutral icons")
            set(_agent_counter_icons "")
            break()
        endif()
    endforeach()
endif()
if(_agent_counter_icons)
    message(STATUS "AgentCounter: provider icons from ${AGENT_COUNTER_ICON_DIR}")
    qt_add_resources(AgentCounter agent_icons
        PREFIX "/agents"
        BASE "${AGENT_COUNTER_ICON_DIR}"
        FILES ${_agent_counter_icons}
    )
    # Tells the selftest the embedded icons are expected, so a resource that
    # failed to embed is a FAIL rather than a silent fallback.
    target_compile_definitions(AgentCounter PRIVATE AGENT_COUNTER_PROVIDER_ICONS=1)
else()
    message(STATUS "AgentCounter: neutral provider icons")
endif()

if(WIN32)
    # NOMINMAX is not tidiness: windows.h defines min/max as macros, and the
    # std::max/std::clamp in groupAgents fail to compile without it.
    target_compile_definitions(AgentCounter PRIVATE
        UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(MSVC)
    target_compile_options(AgentCounter PRIVATE /MP /utf-8)
endif()
