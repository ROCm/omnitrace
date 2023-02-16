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
from .parser import parseFiles
from .parser import parseUploadedFile
from .parser import getSpeedupData
import plotly.graph_objects as go


file_timestamp = 0
data = pd.DataFrame()
samples = pd.DataFrame()
input_filters = None
checklist_options = None
checklist_values = None
workload_path = ""
pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning


IS_DARK = True  # default dark theme


def build_line_graph(data, KernelName, numPoints):
    data_options = sorted(list(set(data.point)))
    layout1 = html.Div(
        id="graph_all",
        className="graph_all",
        children=[html.H4("All Causal Profiles", style={"color": "white"})],
    )

    layout2 = html.Div(
        id="graph_select",
        className="graph_select",
        children=[
            html.H4("Call Stack Sample Histogram", style={"color": "white"}),
        ],
    )

    return layout1, layout2


def update_line_graph(sortFilter, func_list, exp_list, data, numPoints, samples):
    # df = px.data.gapminder() # replace with your own data source
    if "Alphabetical" in sortFilter:
        data = data.sort_values(by=["point", "idx"])

    if "Impact" in sortFilter:
        data = data.sort_values(by=["impact avg", "idx"], ascending=False)

    if "Max Speedup" in sortFilter:
        data = data.sort_values(by=["Max Speedup", "idx"])

    if "Min Speedup" in sortFilter:
        data = data.sort_values(by=["Min Speedup", "idx"])

    if numPoints > 0:
        data = data[data["point count"] > numPoints]

    mask_all = data[data.point.isin(func_list)]
    mask_all = mask_all[mask_all["progress points"].isin(exp_list)]

    mask_select = data[data.point.isin(func_list)]
    mask_select = mask_select[mask_select["progress points"].isin(exp_list)]

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

    causalLayout = [
        html.H4("Selected Causal Profiles", style={"color": "white"}),
    ]

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


def reset_Input_filters(max_points):
    sortOptions = ["Alphabetical", "Max Speedup", "Min Speedup", "Impact"]

    input_filters = [
        {
            "Name": "Sort by",
            "filter": [],
            "values": list(map(str, sortOptions)),
            "type": "Name",
        },
        {"Name": "points", "filter": [], "values": max_points - 1, "type": "int"},
    ]
    return input_filters


def build_causal_layout(
    app,
    runs,
    input_filters_,
    path_to_dir,
    data_,
    samples_,
    verbose=0,
):
    """
    Build gui layout
    """
    global data
    global samples
    data = data_
    samples = samples_
    global input_filters
    input_filters = input_filters_
    global workload_path
    workload_path = path_to_dir
    program_names = sorted(list(set(data.point)))

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

    app.layout = html.Div(style={"backgroundColor": "rgb(50, 50, 50)" if IS_DARK else ""})

    filt_kernel_names = []
    line_graph1, line_graph2 = build_line_graph(
        data, filt_kernel_names, inital_min_points
    )
    app.layout.children = html.Div(
        children=[
            get_header(
                runs[path_to_dir], dropDownMenuItems, input_filters, filt_kernel_names
            ),
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
        sortFilter,
        funcRegex,
        expRegex,
        numPoints,
        workloadPath,
        contentsList,
        filename,
        divChildren,
    ):
        global file_timestamp
        global data
        global input_filters
        global workload_path
        CLI = False

        # change to if debug
        if True:
            print("Sort by is ", sortFilter)
            print("funcRegex is ", funcRegex)
            print("expRegex is ", expRegex)
            print("points is: ", numPoints)
            print("workload_path is: ", workload_path)

        divChildren = []
        files = []
        fig1 = None
        fig2 = None
        global new_data
        global func_list
        global exp_list
        if workloadPath != None:
            print("workload_path:" + workload_path)
            print("workloadPath: " + workloadPath)
        if workloadPath is not None and os.path.exists(workloadPath):
            # files = glob.glob(os.path.join(workload_path, "*.coz")) +
            files = []
            if os.path.isfile(workloadPath):
                files.append(workloadPath)
                workload_path = workloadPath
                print("workload_path:" + workload_path)
            elif os.path.isdir(workloadPath):
                _files = glob.glob(os.path.join(workloadPath, "*.json"))
                # subfiles = glob.glob(os.path.join(workload_path, "*/*.coz")) +
                subfiles = glob.glob(os.path.join(workloadPath, "*/*.json"))
                metadata = glob.glob(os.path.join(workloadPath, "*/metadata*.json"))
                files = _files + subfiles
                # assuming only one file for now
                workload_path = files[0]
                print("workload_path:" + workload_path)

            print("all_files: ")
            print(files)
            # for profile_path in all_files:
            data = parseFiles(files)

            # reset checklists
            func_list = sorted(list(data.point.unique()))
            exp_list = sorted(list(data["progress points"].unique()))

            max_points = data.point.value_counts().max().max()

            # reset input_filters
            input_filters = reset_Input_filters(max_points)

            screen_data, fig1, fig2 = update_line_graph(
                sortFilter, func_list, exp_list, new_data, numPoints, samples
            )

            header = get_header(data, dropDownMenuItems, input_filters, filt_kernel_names)
            return (divChildren, header, fig1, fig2)

        elif contentsList is not None:
            if ".coz" in filename or ".json" in filename:
                new_data_file = base64.decodebytes(
                    contentsList.encode("utf-8").split(b";base64,")[1]
                ).decode("utf-8")

                new_data = parseUploadedFile(new_data_file)
                data = new_data

                max_points = new_data.point.value_counts().max().max()

                func_list = sorted(list(data.point.unique()))
                exp_list = sorted(list(data["progress points"].unique()))

                # reset input_filters
                input_filters = reset_Input_filters(max_points)

                screen_data, fig1, fig2 = update_line_graph(
                    sortFilter, func_list, exp_list, new_data, numPoints, samples
                )
                header = get_header(
                    data, dropDownMenuItems, input_filters, filt_kernel_names
                )
                return (divChildren, header, fig1, fig2)

        # runs when function or experiment regex is changed, takes numPoints into account as well
        elif (
            funcRegex is not None or expRegex is not None
        ):  # or funcRegex != "" or expRegex != "":
            # filter options and values
            if funcRegex is not None:
                p = re.compile(funcRegex, flags=0)

                func_list = [s for s in list(data["point"].unique()) if p.match(s)]
                print(func_list)
            if expRegex is not None:
                p = re.compile(expRegex, flags=0)

                exp_list = [
                    s for s in list(data["progress points"].unique()) if p.match(s)
                ]
                print(exp_list)

            # change to update checklist after points selection
            screen_data, fig1, fig2 = update_line_graph(
                sortFilter,
                func_list,
                exp_list,
                data,
                numPoints,
                samples,
            )

            return (divChildren, header, fig1, fig2)

        # runs when min points changed and when page is first loaded
        if "refresh" == ctx.triggered_id:
            print("refreshing Data with " + workload_path)

            data = parseFiles([workload_path])

            func_list = sorted(list(data.point.unique()))
            exp_list = sorted(list(data["progress points"].unique()))

            screen_data, fig1, fig2 = update_line_graph(
                sortFilter, func_list, exp_list, data, numPoints
            )

            return (divChildren, header, fig1, fig2)
        else:
            func_list = sorted(list(data.point.unique()))
            exp_list = sorted(list(data["progress points"].unique()))

            screen_data, fig1, fig2 = update_line_graph(
                sortFilter,
                func_list,
                exp_list,
                data,
                numPoints,
                samples,
            )

            return (divChildren, header, fig1, fig2)
