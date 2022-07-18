// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "library/components/category_region.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"

#include <timemory/timemory.hpp>

#include <rccl.h>

#include <dlfcn.h>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

TIMEMORY_DECLARE_COMPONENT(rccl_comm_data)

#if !defined(NUM_TIMEMORY_RCCLP_WRAPPERS)
#    define NUM_TIMEMORY_RCCLP_WRAPPERS 15
#endif

namespace tim
{
namespace component
{
template <typename Toolset, typename Tag>
struct rcclp_handle;
}
}  // namespace tim

struct rcclp_tag
{};

using api_t               = rcclp_tag;
using rccl_data_tracker_t = tim::component::data_tracker<float, api_t>;

TIMEMORY_STATISTICS_TYPE(rccl_data_tracker_t, float)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, rccl_data_tracker_t, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, rccl_data_tracker_t, true_type)

using rccl_toolset_t =
    tim::component_bundle<TIMEMORY_API, omnitrace::component::category_region<tim::category::rccl>,
                         tim::component::rccl_comm_data*>;
using rcclp_handle_t           = omnitrace::comp::rcclp_handle<rccl_toolset_t, api_t>;
static uint64_t global_id      = std::numeric_limits<uint64_t>::max();
static void*    librccl_handle = nullptr;

namespace tim
{
namespace component
{
template <typename Toolset, typename Tag>
struct rcclp_handle : base<rcclp_handle<Toolset, Tag>, void>
{
    static constexpr size_t rcclp_wrapper_count = NUM_TIMEMORY_RCCLP_WRAPPERS;

    using value_type = void;
    using this_type  = rcclp_handle<Toolset, Tag>;
    using base_type  = base<this_type, value_type>;

    using string_t       = std::string;
    using nccl_toolset_t = Toolset;
    using rcclp_gotcha_t =
        tim::component::gotcha<rcclp_wrapper_count, nccl_toolset_t, Tag>;
    using rcclp_tuple_t = tim::component_tuple<rcclp_gotcha_t>;
    using toolset_ptr_t = std::shared_ptr<rcclp_tuple_t>;

    static string_t label() { return "rcclp_handle"; }
    static string_t description() { return "Handle for activating NCCL wrappers"; }

    void get() {}

    void start()
    {
        if(get_tool_count()++ == 0)
        {
            get_tool_instance() = std::make_shared<rcclp_tuple_t>("timemory_rcclp");
            get_tool_instance()->start();
        }
    }

    void stop()
    {
        auto idx = --get_tool_count();
        if(get_tool_instance().get())
        {
            get_tool_instance()->stop();
            if(idx == 0) get_tool_instance().reset();
        }
    }

    int get_count() { return get_tool_count().load(); }

private:
    struct persistent_data
    {
        std::atomic<short>   m_configured;
        std::atomic<int64_t> m_count;
        toolset_ptr_t        m_tool;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    static std::atomic<short>& get_configured()
    {
        return get_persistent_data().m_configured;
    }

    static toolset_ptr_t& get_tool_instance() { return get_persistent_data().m_tool; }

    static std::atomic<int64_t>& get_tool_count()
    {
        return get_persistent_data().m_count;
    }
};

template <typename Toolset, typename Tag>
static uint64_t
activate_rcclp()
{
    using handle_t = tim::component::rcclp_handle<Toolset, Tag>;

    static std::shared_ptr<handle_t> _handle;

    if(!_handle.get())
    {
        _handle = std::make_shared<handle_t>();
        _handle->start();

        auto cleanup_functor = [=]() {
            if(_handle)
            {
                _handle->stop();
                _handle.reset();
            }
        };

        std::stringstream ss;
        ss << "timemory-rcclp-" << typeid(Toolset).name() << "-" << typeid(Tag).name();
        tim::manager::instance()->add_cleanup(ss.str(), cleanup_functor);
        return 1;
    }
    return 0;
}
//
//======================================================================================//
//
/// \fn uint64_t tim::component::deactivate_rcclp(uint64_t id)
/// \brief The thread that created the initial rcclp handle will turn off. Returns
/// the number of handles active
///
template <typename Toolset, typename Tag>
static uint64_t
deactivate_rcclp(uint64_t id)
{
    if(id > 0)
    {
        std::stringstream ss;
        ss << "timemory-rcclp-" << typeid(Toolset).name() << "-" << typeid(Tag).name();
        tim::manager::instance()->cleanup(ss.str());
        return 0;
    }
    return 1;
}

//
template <typename Toolset, typename Tag>
void
configure_rcclp(const std::set<std::string>& permit = {},
                const std::set<std::string>& reject = {})
{
    static constexpr size_t rcclp_wrapper_count = NUM_TIMEMORY_RCCLP_WRAPPERS;

    using string_t       = std::string;
    using rcclp_gotcha_t = tim::component::gotcha<rcclp_wrapper_count, Toolset, Tag>;

    static bool is_initialized = false;
    if(!is_initialized)
    {
        // generate the gotcha wrappers
        rcclp_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 0, ncclReduce);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 1, ncclBcast);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 2, ncclBroadcast);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 3, ncclAllReduce);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 4, ncclReduceScatter);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 5, ncclAllGather);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 6, ncclCommCuDevice);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 7, ncclCommUserRank);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 8, ncclGroupStart);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 9, ncclGroupEnd);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 10, ncclSend);
            TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 11, ncclRecv);
            // TIMEMORY_C_GOTCHA(rcclp_gotcha_t, 12, ncclCommCount);
        };

        // provide environment variable for suppressing wrappers
        rcclp_gotcha_t::get_reject_list() = [reject]() {
            auto _reject = reject;
            // check environment
            auto reject_list = tim::get_env<string_t>("OMNITRACE_RCCLP_REJECT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(reject_list))
                _reject.insert(itr);
            return _reject;
        };

        // provide environment variable for selecting wrappers
        rcclp_gotcha_t::get_permit_list() = [permit]() {
            auto _permit = permit;
            // check environment
            auto permit_list = tim::get_env<string_t>("OMNITRACE_RCCLP_PERMIT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(permit_list))
                _permit.insert(itr);
            return _permit;
        };

        is_initialized = true;
    }
}
}  // namespace component
}  // namespace tim

namespace omnitrace
{
namespace rcclp
{
void
configure()
{
    rccl_data_tracker_t::label()       = "rccl_comm_data";
    rccl_data_tracker_t::description() = "Tracks RCCL communication data";
}

void
setup()
{
    configure();

    // make sure the symbols are loaded to be wrapped
    auto libpath   = tim::get_env<std::string>("OMNITRACE_RCCL_LIBRARY", "librccl.so");
    librccl_handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if(!librccl_handle) fprintf(stderr, "%s\n", dlerror());
    dlerror();  // Clear any existing error

    auto _data = tim::get_env("OMNITRACE_RCCLP_COMM_DATA", true);
    if(_data)
        rccl_toolset_t::get_initializer() = [](rccl_toolset_t& cb) {
            cb.initialize<comp::rccl_comm_data>();
        };

    comp::configure_rcclp<rccl_toolset_t, api_t>();
    global_id = comp::activate_rcclp<rccl_toolset_t, api_t>();
    if(librccl_handle) dlclose(librccl_handle);
}

void
shutdown()
{
    if(global_id < std::numeric_limits<uint64_t>::max())
        comp::deactivate_rcclp<rccl_toolset_t, api_t>(global_id);
}
}  // namespace rcclp
}  // namespace omnitrace
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
struct rccl_comm_data : base<rccl_comm_data, void>
{
    using value_type = void;
    using this_type  = rccl_comm_data;
    using base_type  = base<this_type, value_type>;
    using tracker_t  = tim::auto_tuple<rccl_data_tracker_t>;
    using data_type  = float;

    TIMEMORY_DEFAULT_OBJECT(rccl_comm_data)

    static void preinit() { omnitrace::rcclp::configure(); }

    void start() {}
    void stop() {}

    static auto rccl_type_size(ncclDataType_t datatype)
    {
        switch(datatype)
        {
            case ncclInt8:
            case ncclUint8: return 1;
            case ncclFloat16: return 2;
            case ncclInt32:
            case ncclUint32:
            case ncclFloat32: return 4;
            case ncclInt64:
            case ncclUint64:
            case ncclFloat64: return 8;
            default: return 0;
        };
    }

    // ncclReduce
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, ncclRedOp_t, int root, ncclComm_t, hipStream_t)
    {
        int size = rccl_type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // ncclSend
    void audit(const std::string& _name, const void*, size_t count,
               ncclDataType_t datatype, int peer, ncclComm_t, hipStream_t)
    {
        int size = rccl_type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", peer));
    }

    // ncclBcast
    // ncclRecv
    void audit(const std::string& _name, void*, size_t count, ncclDataType_t datatype,
               int root, ncclComm_t, hipStream_t)
    {
        int size = rccl_type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // ncclBroadcast
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, int root, ncclComm_t, hipStream_t)
    {
        int size = rccl_type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // ncclAllReduce
    // ncclReduceScatter
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, ncclRedOp_t, ncclComm_t, hipStream_t)
    {
        int size = rccl_type_size(datatype);
        add(_name, count * size);
    }

    // ncclAllGather
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, ncclComm_t, hipStream_t)
    {
        int size = rccl_type_size(datatype);
        add(_name, count * size);
    }

private:
    template <typename... Args>
    void add(tracker_t& _t, data_type value, Args&&... args)
    {
        _t.store(std::plus<data_type>{}, value);
        TIMEMORY_FOLD_EXPRESSION(add_secondary(_t, std::forward<Args>(args), value));
    }

    template <typename... Args>
    void add(const std::string& _name, data_type value, Args&&... args)
    {
        tracker_t _t(_name);
        add(_t, value, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void add_secondary(tracker_t&, const std::string& _name, data_type value,
                       Args&&... args)
    {
        // if(tim::settings::add_secondary())
        {
            tracker_t _s(_name);
            add(_s, value, std::forward<Args>(args)...);
        }
    }
};
}  // namespace component
}  // namespace tim

TIMEMORY_INITIALIZE_STORAGE(rccl_comm_data, rccl_data_tracker_t)
