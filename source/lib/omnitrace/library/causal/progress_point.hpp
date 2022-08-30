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

#include "library/common.hpp"
#include "library/components/fwd.hpp"
#include "library/defines.hpp"
#include "timemory/components/timing/wall_clock.hpp"

#include <timemory/components/base.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal.hpp>

#include <cstdint>
#include <map>

namespace omnitrace
{
namespace causal
{
struct progress_point
: comp::empty_base
, concepts::component
{
    using value_type = void;
    using hash_type  = tim::hash_value_t;

    static std::string label();
    static std::string description();

    TIMEMORY_DEFAULT_OBJECT(progress_point)

    void start() const;
    void stop() const;
    void set_prefix(hash_type _v);

    struct result
    {
        double throughput = 0;
        double latency    = 0;

        template <typename ArchiveT>
        void serialize(ArchiveT&);
    };

    // the result per function
    using result_data_t = std::map<hash_type, result>;

    static void          setup();  // reset data before experiment
    static result_data_t get();    // compute the experiment result

private:
    hash_type        m_hash       = 0;
    comp::wall_clock m_throughput = {};
};

template <typename ArchiveT>
inline void
progress_point::result::serialize(ArchiveT& ar)
{
    ar(tim::cereal::make_nvp("throughput", throughput),
       tim::cereal::make_nvp("latency", latency));
}
}  // namespace causal
}  // namespace omnitrace
