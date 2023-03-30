#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import

__author__ = "AMD Research"
__copyright__ = "Copyright 2023, Advanced Micro Devices, Inc."
__license__ = "MIT"
__maintainer__ = "AMD Research"
__status__ = "Development"

import glob
import re
import sys
import copy
import base64
import os
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px

from re import L
from selectors import EpollSelector
from matplotlib.axis import XAxis
from numpy import append
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme, Symbol
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from dash import dcc, ctx
from os.path import exists

from .header import get_header
from .parser import parse_files
from .parser import parse_uploaded_file
from .parser import find_causal_files
import plotly.graph_objects as go


file_timestamp = 0
global_data = pd.DataFrame()
global_samples = pd.DataFrame()
global_input_filters = None
workload_path = ""
verbose = 0

pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning


def print_speedup_info(data):
    for idx in set(list(data.idx)):
        sub_data = data[data["idx"] == idx]
        print("")
        for sub_idx, row in sub_data.iterrows():
            print(
                f"""[{row["point"]}][{row["progress points"]}][{row["line speedup"]}] speedup: {row["program speedup"]:6.1f} +/- {row["speedup err"]:6.2f}%"""
            )
        print(
            f"""[{row["point"]}][{row["progress points"]}][sum] impact: {row["impact sum"]:6.1f}"""
        )
        print(
            f"""[{row["point"]}][{row["progress points"]}][avg] impact: {row["impact avg"]:6.1f} +/- {row["impact err"]:6.2f}"""
        )


def build_line_graph():
    layout1 = html.Div(
        id="graph_all",
        className="graph_all",
        children=[html.H4("All Causal Profiles", style={"color": "white"})],
    )

    layout2 = html.Div(
        id="graph_select",
        className="graph_select",
        children=[html.H4("Call Stack Sample Histogram", style={"color": "white"})],
    )

    return layout1, layout2


def update_line_graph(
    sort_filter, func_list, exp_list, data, num_points, samples, workloads
):
    if "Alphabetical" in sort_filter:
        data = data.sort_values(by=["point", "idx"])

    elif "Impact" in sort_filter:
        data = data.sort_values(by=["impact sum", "idx"], ascending=False)

    elif "Max Speedup" in sort_filter:
        data = data.sort_values(by=["max speedup", "idx"])

    elif "Min Speedup" in sort_filter:
        data = data.sort_values(by=["min speedup", "idx"])

    if num_points > 0:
        data = data[data["point count"] > num_points]

    mask = data[data.point.isin(func_list)]
    mask = mask[mask["progress points"].isin(exp_list)]
    mask = mask[mask["workload"].isin(workloads)]

    progress_points = list(mask["progress points"].unique())
    colors = [x / float(len(progress_points)) for x in range(len(progress_points))]
    colors_df = pd.DataFrame()
    for color, prog in zip(colors, progress_points):
        colors_df = pd.concat(
            [colors_df, pd.DataFrame({"progress points": [prog], "color": [color * 100]})]
        )

    causalLayout = [html.H4("Selected Causal Profiles", style={"color": "white"})]

    for point in list(mask.point.unique()):
        subplots = go.Figure()
        sub_data = mask[mask["point"] == point]
        x_label = (
            "Line Speedup"
            if re.match(".*:([0-9]+)$", point) is not None
            else "Function Speedup"
        )
        for prog in list(sub_data["progress points"].unique()):
            sub_data_prog = sub_data[sub_data["progress points"] == prog]
            subplots.add_trace(
                go.Scatter(
                    x=sub_data_prog["line speedup"],
                    y=sub_data_prog["program speedup"],
                    error_y=dict(
                        type="percent", array=sub_data_prog["impact err"].tolist()
                    ),
                    line_shape="spline",
                    name=prog,
                    mode="lines+markers",
                )
            ).update_layout(xaxis={"title": x_label}, yaxis={"title": "Program Speedup"})
        causalLayout.append(html.H4(point, style={"color": "white"}))
        causalLayout.append(dcc.Graph(figure=subplots))

    fig3 = px.bar(samples, x="location", y="count", height=1200)

    samplesLayout = [
        html.H4("Call Stack Sample Histogram", style={"color": "white"}),
        dcc.Graph(figure=fig3),
    ]

    return mask, causalLayout, samplesLayout


def reset_input_filters(workloads, max_points, verbosity):
    sortOptions = ["Alphabetical", "Max Speedup", "Min Speedup", "Impact"]
    if type(workloads) == str:
        workloads = [workloads]

    input_filters = [
        {
            "Name": "Sort by",
            "values": list(map(str, sortOptions)),
            "default": "Impact",
            "type": "Name",
            "multi": False,
        },
        {
            "Name": "Select Workload",
            "values": workloads,
            "default": workloads,
            "type": "Name",
            "multi": True,
        },
        {"Name": "points", "filter": [], "values": max_points, "type": "int"},
    ]
    return input_filters


def build_causal_layout(
    app, input_filters, path_to_dir, data, samples, verbosity=0, light_mode=True
):
    """
    Build gui layout
    """
    global verbose
    global global_data
    global global_samples
    global_data = data
    global_samples = samples
    global global_input_filters
    global_input_filters = input_filters
    global workload_path
    workload_path = path_to_dir

    verbose = verbosity

    dropDownMenuItems = [
        dbc.DropdownMenuItem("Overview", header=True),
        dbc.DropdownMenuItem("All Causal Profiles"),
        dbc.DropdownMenuItem("Selected Causal Profiles"),
    ]

    inital_min_points = 3

    app.layout = html.Div(
        style={"backgroundColor": "rgb(255, 255, 255)"}
        if light_mode
        else {"backgroundColor": "rgb(50, 50, 50)"}
    )

    line_graph1, line_graph2 = build_line_graph()

    if verbosity == 3:
        print(global_input_filters)
    app.layout.children = html.Div(
        children=[
            get_header(dropDownMenuItems, global_input_filters),
            html.Div(id="container", children=[]),
            line_graph1,
            line_graph2,
        ]
    )

    @app.callback(
        Output("container", "children"),
        Output("nav-wrap", "children"),
        Output("graph_all", "children"),
        Output("graph_select", "children"),
        [Input("nav-wrap", "children")],
        [Input("refresh", "n_clicks")],
        [Input("Sort by-filt", "value")],
        [Input("Select Workload-filt", "value")],
        [Input("function_regex", "value")],
        [Input("exp_regex", "value")],
        [Input("points-filt", "value")],
        [Input("file-path", "value")],
        [Input("upload-drag", "contents")],
        [State("upload-drag", "filename")],
        [State("container", "children")],
    )
    def generate_from_filter(
        header,
        refresh,
        sort_filter,
        workload_filter,
        func_regex,
        exp_regex,
        num_points,
        _workload_path,
        contentsList,
        filename,
        divChildren,
    ):
        global file_timestamp
        global global_data
        global global_input_filters
        global workload_path

        # change to if debug
        if verbose >= 3:
            print("Sort by is ", sort_filter)
            print("funcRegex is ", func_regex)
            print("expRegex is ", exp_regex)
            print("points is: ", num_points)
            print("workload_path is: ", workload_path)
            print("selected workloads are:", workload_filter)

        divChildren = []
        files = []
        fig1 = None
        fig2 = None
        global new_data
        global func_list
        global exp_list
        global global_samples
        screen_data = pd.DataFrame()

        if _workload_path is not None and os.path.exists(_workload_path):
            files = []
            if os.path.isfile(_workload_path):
                files.append(_workload_path)
                workload_path = [_workload_path]
            elif os.path.isdir(_workload_path):
                _files = glob.glob(os.path.join(_workload_path, "*.json"))
                # subfiles = glob.glob(os.path.join(workload_path, "*/*.coz")) +
                subfiles = glob.glob(os.path.join(_workload_path, "*/*.json"))
                metadata = glob.glob(os.path.join(_workload_path, "*/metadata*.json"))
                files = _files + subfiles
                workload_path = files
            global_data, global_samples, global_filenames = parse_files(
                workload_path, verbose=verbose
            )
            func_list = sorted(list(global_data.point.unique()))
            exp_list = sorted(list(global_data["progress points"].unique()))

            max_points = global_data.point.value_counts().max().max()

            # reset input_filters
            workloads = [os.path.basename(file) for file in files]
            global_input_filters = reset_input_filters(workloads, max_points, verbose)

            screen_data, fig1, fig2 = update_line_graph(
                sort_filter,
                func_list,
                exp_list,
                global_data,
                num_points,
                global_samples,
                workloads,
            )

            header = get_header(dropDownMenuItems, global_input_filters)

        elif contentsList is not None:
            if ".coz" in filename or ".json" in filename:
                new_data_file = base64.decodebytes(
                    contentsList.encode("utf-8").split(b";base64,")[1]
                ).decode("utf-8")

                new_data, global_samples = parse_uploaded_file(filename, new_data_file)
                global_data = new_data

                max_points = new_data.point.value_counts().max().max()

                func_list = sorted(list(global_data.point.unique()))
                exp_list = sorted(list(global_data["progress points"].unique()))

                # reset input_filters
                global_input_filters = reset_input_filters(
                    [filename], max_points, verbose
                )

                screen_data, fig1, fig2 = update_line_graph(
                    sort_filter,
                    func_list,
                    exp_list,
                    new_data,
                    num_points,
                    global_samples,
                    workload_filter,
                )
                header = get_header(dropDownMenuItems, global_input_filters)

        # runs when function or experiment regex is changed, takes numPoints into account as well
        elif func_regex is not None or exp_regex is not None:
            if global_data.empty == False:
                if func_regex is not None:
                    p = re.compile(func_regex, flags=0)

                    func_list = [
                        s for s in list(global_data["point"].unique()) if p.match(s)
                    ]
                if exp_regex is not None:
                    p = re.compile(exp_regex, flags=0)

                    exp_list = [
                        s
                        for s in list(global_data["progress points"].unique())
                        if p.match(s)
                    ]

                screen_data, fig1, fig2 = update_line_graph(
                    sort_filter,
                    func_list,
                    exp_list,
                    global_data,
                    num_points,
                    global_samples,
                    workload_filter,
                )
        elif "refresh" == ctx.triggered_id:
            if verbose >= 3:
                print("refreshing Data with ", workload_path)

            files = find_causal_files(workload_path, verbose, False)

            if verbose >= 3:
                print("files found", files)

            global_data, global_samples, global_filenames = parse_files(
                files=files, verbose=verbose
            )

            if global_data.empty == False:
                max_points = global_data.point.value_counts().max().max()

                # reset input_filters
                global_input_filters = reset_input_filters(
                    global_filenames, max_points, verbose
                )
                header = get_header(dropDownMenuItems, global_input_filters)
                if verbose >= 3:
                    print(global_data.keys())

                func_list = sorted(list(global_data.point.unique()))
                exp_list = sorted(list(global_data["progress points"].unique()))

                screen_data, fig1, fig2 = update_line_graph(
                    sort_filter,
                    func_list,
                    exp_list,
                    global_data,
                    num_points,
                    global_samples,
                    global_filenames,
                )
        else:
            func_list = []
            exp_list = []
            if global_data.empty == False:
                if verbose == 3:
                    print(global_data)
                func_list = sorted(list(global_data.point.unique()))
                exp_list = sorted(list(global_data["progress points"].unique()))
                screen_data, fig1, fig2 = update_line_graph(
                    sort_filter,
                    func_list,
                    exp_list,
                    global_data,
                    num_points,
                    global_samples,
                    workload_filter,
                )
        if screen_data.empty == False:
            if verbose == 2:
                print_speedup_info(screen_data)
            elif verbose >= 3:
                print(screen_data)

        return (divChildren, header, fig1, fig2)
