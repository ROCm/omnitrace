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

#include "library/components/comm_data.hpp"
#include "library/components/fwd.hpp"

#include <timemory/manager.hpp>
#include <timemory/units.hpp>

namespace tim
{
namespace component
{
void
comm_data::preinit()
{
    configure();
}

void
comm_data::global_finalize()
{
    configure();
}

void
comm_data::configure()
{
    comm_data_tracker_t::label()       = "comm_data";
    comm_data_tracker_t::description() = "Tracks MPI/RCCL communication data sizes";
    comm_data_tracker_t::set_precision(3);
    comm_data_tracker_t::set_unit(units::kilobyte);
    comm_data_tracker_t::set_display_unit("KB");
}

#if defined(OMNITRACE_USE_MPI)
// MPI_Send
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm)
{
    auto _name = std::string_view{ _data.tool_id };
    int  _size = mpi_type_size(datatype);
    if(_size == 0) return;
    tracker_t _t{ _name };
    add(_t, count * _size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * _size,
                  TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
}

// MPI_Recv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Status*)
{
    auto _name = std::string_view{ _data.tool_id };
    int  _size = mpi_type_size(datatype);
    if(_size == 0) return;
    tracker_t _t{ _name };
    add(_t, count * _size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * _size,
                  TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
}

// MPI_Isend
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    auto _name = std::string_view{ _data.tool_id };
    int  _size = mpi_type_size(datatype);
    if(_size == 0) return;
    tracker_t _t{ _name };
    add(_t, count * _size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * _size,
                  TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
}

// MPI_Irecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int dst, int tag, MPI_Comm, MPI_Request*)
{
    auto _name = std::string_view{ _data.tool_id };
    int  _size = mpi_type_size(datatype);
    if(_size == 0) return;
    tracker_t _t{ _name };
    add(_t, count * _size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * _size,
                  TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
}

// MPI_Bcast
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, int count,
                 MPI_Datatype datatype, int root, MPI_Comm)
{
    auto _name = std::string_view{ _data.tool_id };
    int  _size = mpi_type_size(datatype);
    if(_size == 0) return;
    add(_name, count * _size, TIMEMORY_JOIN('_', _name, "root", root));
}

// MPI_Allreduce
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*, int count,
                 MPI_Datatype datatype, MPI_Op, MPI_Comm)
{
    auto _name = std::string_view{ _data.tool_id };
    int  _size = mpi_type_size(datatype);
    if(_size == 0) return;
    add(_name, count * _size);
}

// MPI_Sendrecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, int, int sendtag, void*, int recvcount,
                 MPI_Datatype recvtype, int, int recvtag, MPI_Comm, MPI_Status*)
{
    auto _name      = std::string_view{ _data.tool_id };
    int  _send_size = mpi_type_size(sendtype);
    int  _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;
    tracker_t _t{ _name };
    add(_t, sendcount * _send_size + recvcount * _recv_size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "send"), sendcount * _send_size,
                  TIMEMORY_JOIN('_', _name, "send", "tag", sendtag));
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "recv"), recvcount * _recv_size,
                  TIMEMORY_JOIN('_', _name, "recv", "tag", recvtag));
}

// MPI_Gather
// MPI_Scatter
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm)
{
    auto _name      = std::string_view{ _data.tool_id };
    int  _send_size = mpi_type_size(sendtype);
    int  _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;
    tracker_t _t{ _name };
    add(_t, sendcount * _send_size + recvcount * _recv_size);
    tracker_t _r(TIMEMORY_JOIN('_', _name, "root", root));
    add(_r, sendcount * _send_size + recvcount * _recv_size);
    add_secondary(_r, TIMEMORY_JOIN('_', _name, "root", root, "send"),
                  sendcount * _send_size);
    add_secondary(_r, TIMEMORY_JOIN('_', _name, "root", root, "recv"),
                  recvcount * _recv_size);
}

// MPI_Alltoall
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, int sendcount,
                 MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm)
{
    auto _name      = std::string_view{ _data.tool_id };
    int  _send_size = mpi_type_size(sendtype);
    int  _recv_size = mpi_type_size(recvtype);
    if(_send_size == 0 || _recv_size == 0) return;
    tracker_t _t{ _name };
    add(_t, sendcount * _send_size + recvcount * _recv_size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "send"), sendcount * _send_size);
    add_secondary(_t, TIMEMORY_JOIN('_', _name, "recv"), recvcount * _recv_size);
}
#endif

#if defined(OMNITRACE_USE_RCCL)
// ncclReduce
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                 size_t count, ncclDataType_t datatype, ncclRedOp_t, int root, ncclComm_t,
                 hipStream_t)
{
    int _size = rccl_type_size(datatype);
    add(_data, count * _size, JOIN('_', _data.tool_id.c_str(), "root", root));
}

// ncclSend
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, size_t count,
                 ncclDataType_t datatype, int peer, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    add(_data, count * _size, JOIN('_', _data.tool_id.c_str(), "root", peer));
}

// ncclBcast
// ncclRecv
void
comm_data::audit(const gotcha_data& _data, audit::incoming, void*, size_t count,
                 ncclDataType_t datatype, int root, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    add(_data, count * _size, JOIN('_', _data.tool_id.c_str(), "root", root));
}

// ncclBroadcast
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                 size_t count, ncclDataType_t datatype, int root, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    add(_data, count * _size, JOIN('_', _data.tool_id.c_str(), "root", root));
}

// ncclAllReduce
// ncclReduceScatter
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                 size_t count, ncclDataType_t datatype, ncclRedOp_t, ncclComm_t,
                 hipStream_t)
{
    int _size = rccl_type_size(datatype);
    add(_data, count * _size);
}

// ncclAllGather
void
comm_data::audit(const gotcha_data& _data, audit::incoming, const void*, void*,
                 size_t count, ncclDataType_t datatype, ncclComm_t, hipStream_t)
{
    int _size = rccl_type_size(datatype);
    add(_data, count * _size);
}
#endif
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<float, tim::api::omnitrace>), true, float)

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(comm_data, false, void)
