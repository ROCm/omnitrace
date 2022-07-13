// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/rocprofiler.hpp"

#include <timemory/api.hpp>
#include <timemory/components/base.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/enum.h>
#include <timemory/macros/os.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/transient_function.hpp>

namespace tim
{
namespace component
{
using rocprofiler_value = typename omnitrace::rocprofiler::rocm_event::value_type;
using rocprofiler_data  = data_tracker<rocprofiler_value, rocprofiler>;

struct rocprofiler
: base<rocprofiler, void>
, private policy::instance_tracker<rocprofiler, false>
{
    using value_type   = void;
    using base_type    = base<rocprofiler, void>;
    using tracker_type = policy::instance_tracker<rocprofiler, false>;

    TIMEMORY_DEFAULT_OBJECT(rocprofiler)

    static void preinit();
    static void global_init() { setup(); }
    static void global_finalize() { shutdown(); }

    static bool is_setup();
    static void setup();
    static void shutdown();
    static void add_setup(const std::string&, std::function<void()>&&);
    static void add_shutdown(const std::string&, std::function<void()>&&);
    static void remove_setup(const std::string&);
    static void remove_shutdown(const std::string&);

    void start();
    void stop();

    // this function protects rocprofiler_flush_activty from being called
    // when omnitrace exits during a callback
    [[nodiscard]] static scope::transient_destructor protect_flush_activity();
};

#if !defined(OMNITRACE_USE_ROCTRACER)
inline void
rocprofiler::setup()
{}

inline void
rocprofiler::shutdown()
{}

inline bool
rocprofiler::is_setup()
{
    return false;
}
#endif
}  // namespace component
}  // namespace tim

#if !defined(OMNITRACE_USE_ROCTRACER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::rocprofiler_data, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::rocprofiler, false_type)
#endif

TIMEMORY_SET_COMPONENT_API(component::rocprofiler_data, project::timemory,
                           category::timing, os::supports_unix)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::rocprofiler_data,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::rocprofiler_data, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::rocprofiler_data, false_type)
TIMEMORY_STATISTICS_TYPE(component::rocprofiler_data, component::rocprofiler_value)

#if !defined(OMNITRACE_EXTERN_COMPONENTS) ||                                             \
    (defined(OMNITRACE_EXTERN_COMPONENTS) && OMNITRACE_EXTERN_COMPONENTS > 0)

#    include <timemory/operations.hpp>

TIMEMORY_DECLARE_EXTERN_COMPONENT(rocprofiler, false, void)
TIMEMORY_DECLARE_EXTERN_COMPONENT(rocprofiler_data, true, double)

#endif
