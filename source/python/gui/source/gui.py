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
import plotly.graph_objects as go


file_timestamp = 0
global_data = pd.DataFrame()
global_samples = pd.DataFrame()
global_input_filters = None
workload_path = ""
pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning


def build_line_graph():
    # data_options = sorted(list(set(data.point)))
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
    # df = px.data.gapminder() # replace with your own data source
    if "Alphabetical" in sort_filter:
        data = data.sort_values(by=["point", "idx"])

    if "Impact" in sort_filter:
        data = data.sort_values(by=["impact sum", "idx"], ascending=False)

    if "Max Speedup" in sort_filter:
        data = data.sort_values(by=["Max Speedup", "idx"])

    if "Min Speedup" in sort_filter:
        data = data.sort_values(by=["Min Speedup", "idx"])

    if num_points > 0:
        data = data[data["point count"] > num_points]

    mask_all = data[data.point.isin(func_list)]
    mask_all = mask_all[mask_all["progress points"].isin(exp_list)]
    mask_all = mask_all[mask_all["workload"].isin(workloads)]

    mask_select = data[data.point.isin(func_list)]
    mask_select = mask_select[mask_select["progress points"].isin(exp_list)]
    mask_select = mask_select[mask_select["workload"].isin(workloads)]

    progress_points = list(mask_select["progress points"].unique())
    colors = [x / float(len(progress_points)) for x in range(len(progress_points))]
    # colors = [0 if x == 0 else 1/float(x) for x in colors]
    colors_df = pd.DataFrame()
    for color, prog in zip(colors, progress_points):
        colors_df = pd.concat(
            [colors_df, pd.DataFrame({"progress points": [prog], "color": [color * 100]})]
        )

    fig1 = go.Figure()

    for point in sorted(list(mask_all.point.unique())):
        # for experiment in list(mask_select.experiment)[0:3]:
        sub_data = mask_all[mask_all["point"] == point]
        fig1.add_trace(
            go.Scatter(
                x=sub_data["Line Speedup"],
                y=sub_data["Program Speedup"],
                line_shape="spline",
                name=point[0:50],
                mode="lines+markers",
            )
        ).update_layout(
            xaxis={"title": "Function Speedup"}, yaxis={"title": "Program Speedup"}
        )

    causalLayout = [html.H4("Selected Causal Profiles", style={"color": "white"})]

    for point in list(mask_select.point.unique()):
        subplots = go.Figure()
        sub_data = mask_select[mask_select["point"] == point]
        line_number = point[point.rfind(":") :].isnumeric()
        if line_number:
            # untested
            for prog in list(sub_data["progress points"].unique()):
                sub_data_prog = sub_data[sub_data["progress points"] == prog]
                subplots.add_trace(
                    go.Scatter(
                        x=sub_data_prog["Line Speedup"],
                        y=sub_data_prog["Program Speedup"],
                        error_y=dict(
                            type="percent", array=sub_data_prog["impact err"].tolist()
                        ),
                        line_shape="spline",
                        name=prog,
                        mode="lines+markers",
                    )
                ).update_layout(
                    xaxis={"title": "Line Speedup"}, yaxis={"title": "Program Speedup"}
                )
        else:
            for prog in list(sub_data["progress points"].unique()):
                sub_data_prog = sub_data[sub_data["progress points"] == prog]
                subplots.add_trace(
                    go.Scatter(
                        x=sub_data_prog["Line Speedup"],
                        y=sub_data_prog["Program Speedup"],
                        line_color="hsv("
                        + str(colors_df[colors_df["progress points"] == prog]["color"][0])
                        + "%,100%,100%)",
                        error_y=dict(
                            type="percent", array=sub_data_prog["impact err"].tolist()
                        ),
                        line_shape="spline",
                        name=prog,
                        mode="lines+markers",
                    )
                ).update_layout(
                    xaxis={"title": "Function Speedup"},
                    yaxis={"title": "Program Speedup"},
                )
        causalLayout.append(html.H4(point, style={"color": "white"}))
        causalLayout.append(dcc.Graph(figure=subplots))

    fig3 = px.bar(samples, x="location", y="count", height=1200)

    samplesLayout = [
        html.H4("Call Stack Sample Histogram", style={"color": "white"}),
        dcc.Graph(figure=fig3),
    ]

    return mask_all, causalLayout, samplesLayout


def reset_input_filters(workloads, max_points):
    sortOptions = ["Alphabetical", "Max Speedup", "Min Speedup", "Impact"]

    input_filters = [
        {
            "Name": "Sort by",
            "default": sortOptions[0],
            "values": list(map(str, sortOptions)),
            "type": "Name",
        },
        {
            "Name": "Select Workload",
            "values": workloads,
            "default": workloads,
            "type": "Name",
        },
        {"Name": "points", "filter": [], "values": max_points - 1, "type": "int"},
    ]
    return input_filters


def build_causal_layout(
    app, runs, input_filters, path_to_dir, data, samples, verbose=0, light_mode=True
):
    """
    Build gui layout
    """
    global global_data
    global global_samples
    global_data = data
    global_samples = samples
    global global_input_filters
    global_input_filters = input_filters
    global workload_path
    workload_path = path_to_dir

    dropDownMenuItems = [
        dbc.DropdownMenuItem("Overview", header=True),
        dbc.DropdownMenuItem(
            "All Causal Profiles", href="#graph_all", external_link=True
        ),
        dbc.DropdownMenuItem(
            "Selected Causal Profiles", href="#graph_select", external_link=True
        ),
    ]

    inital_min_points = 3

    app.layout = html.Div(
        style={"backgroundColor": "rgb(255, 255, 255)"}
        if light_mode
        else {"backgroundColor": "rgb(50, 50, 50)"}
    )

    filt_kernel_names = []

    line_graph1, line_graph2 = build_line_graph()

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
        CLI = False

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

        if _workload_path is not None and os.path.exists(_workload_path):
            # files = glob.glob(os.path.join(workload_path, "*.coz")) +
            files = []
            if os.path.isfile(_workload_path):
                files.append(_workload_path)
                workload_path = _workload_path
            elif os.path.isdir(_workload_path):
                _files = glob.glob(os.path.join(_workload_path, "*.json"))
                # subfiles = glob.glob(os.path.join(workload_path, "*/*.coz")) +
                subfiles = glob.glob(os.path.join(_workload_path, "*/*.json"))
                metadata = glob.glob(os.path.join(_workload_path, "*/metadata*.json"))
                files = _files + subfiles
                workload_path = files
            global_data, global_samples = parse_files(files)
            # reset checklists
            func_list = sorted(list(global_data.point.unique()))
            exp_list = sorted(list(global_data["progress points"].unique()))

            max_points = global_data.point.value_counts().max().max()

            # reset input_filters
            workloads = [os.path.basename(file) for file in files]
            global_input_filters = reset_input_filters(workloads, max_points)

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
            return (divChildren, header, fig1, fig2)

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
                global_input_filters = reset_input_filters([filename], max_points)

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
                return (divChildren, header, fig1, fig2)

        # runs when function or experiment regex is changed, takes numPoints into account as well
        elif func_regex is not None or exp_regex is not None:
            # filter options and values
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

            return (divChildren, header, fig1, fig2)

        # runs when min points changed and when page is first loaded
        if "refresh" == ctx.triggered_id:
            print("refreshing Data with " + workload_path)

            global_data = parse_files([workload_path])

            func_list = sorted(list(global_data.point.unique()))
            exp_list = sorted(list(global_data["progress points"].unique()))
            if global_data.empty == False:
                screen_data, fig1, fig2 = update_line_graph(
                    sort_filter,
                    func_list,
                    exp_list,
                    global_data,
                    num_points,
                    workload_filter,
                )

            return (divChildren, header, fig1, fig2)
        else:
            func_list = []
            exp_list = []
            if global_data.empty == False:
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

            return (divChildren, header, fig1, fig2)
