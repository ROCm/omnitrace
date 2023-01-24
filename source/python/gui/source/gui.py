################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

from __future__ import absolute_import

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
from dash import dcc
from os.path import exists

from .header import get_header
from .parser import parseFiles
from .parser import parseUploadedFile
from .parser import getSpeedupData
import plotly.graph_objects as go


file_timestamp = 0
data = pd.DataFrame()
input_filters = None
checklist_options = None
checklist_values = None
pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning


IS_DARK = True  # default dark theme


def build_line_graph(data, KernelName, points_filt):
    data_options = sorted(list(set(data.point)))
    layout1 = html.Div(
        id = "graph_all",
        className = 'graph_all',
        children = [
            html.H4("All Causal Profiles", style={"color": "white"}),
        ]
    )

    layout2 = html.Div(
        id = "graph_select",
        className = 'graph_select',
        children = [
            html.H4("Selected Causal Profiles", style={"color": "white"}),
        ]
    )

    return layout1, layout2


def update_line_graph(sort_filt, func_list, exp_list, data, points_filt):

    # df = px.data.gapminder() # replace with your own data source
    if "Alphabetical" in sort_filt:
        data = data.sort_values(by="point")
    if "Impact" in sort_filt:
        newData = pd.DataFrame()
        impactOrder = pd.DataFrame(data.point.unique(), columns=["Program"])

        for index_imp, curr in impactOrder.iterrows():
            prev = pd.Series()
            data_subset = data[data["point"] == curr.Program]
            area = 0
            norm_area = 0
            for index_sub, data_point in data_subset.iterrows():
                if prev.empty:
                    prev = data_point
                else:
                    avg_progress_speedup = (
                        prev["Program Speedup"] + data_point["Program Speedup"]
                    ) / 2
                    area = area + avg_progress_speedup * (
                        data_point["Line Speedup"] - prev["Line Speedup"]
                    )
                    norm_area = area / data_point["Line Speedup"]
                    prev = data_point
            impactOrder.at[index_imp, "area"] = norm_area
        impactOrder = impactOrder.sort_values(by="area")
        impactOrder = impactOrder.Program.unique()

        # add to newData in impact order
        for point in impactOrder:
            data_subset = data[data["point"] == point]
            newData = pd.concat([data_subset, newData])
        data = newData
    if "Max Speedup" in sort_filt:
        speedupOrder = data.sort_values(by="Program Speedup").point.unique()
        newData = pd.DataFrame()
        for point in speedupOrder:
            data_subset = data[data["point"] == point]
            newData = pd.concat([data_subset, newData])
        data = newData
    if "Min Speedup" in sort_filt:
        speedupOrder = data.sort_values(
            by="Program Speedup", ascending=False
        ).point.unique()
        newData = pd.DataFrame()
        for point in speedupOrder:
            data_subset = data[data["point"] == point]
            newData = pd.concat([data_subset, newData])
        data = newData

    point_counts = data.point.value_counts()

    sufficient_points = point_counts > points_filt
    sufficient_points = sufficient_points.loc[lambda x: x == True]
    sufficient_points = list(sufficient_points.index)
    # sufficient_points = sufficient_points
    mask_all = data[data.point.isin(func_list)]
    mask_all = mask_all[mask_all.point.isin(sufficient_points)]
    mask_all = mask_all[mask_all["progress points"].isin(exp_list)]

    mask_select = data[data.point.isin(func_list)]
    mask_select = mask_select[mask_select.point.isin(sufficient_points)]
    mask_select = mask_select[mask_select["progress points"].isin(exp_list)]

    # what = mask_select.value_counts()[True]
    # what = data[mask_all]
    # fig_data1 = data[mask_all]
    # fig_data2 = data[mask_select]
    
    fig1 = go.Figure()

    for point in sorted(list(mask_all.point.unique())):
        #for experiment in list(mask_select.experiment)[0:3]:
        sub_data = mask_all[mask_all["point"] == point]
        fig1.add_trace(
                    go.Scatter(
                    x = sub_data["Line Speedup"],
                    y = sub_data["Program Speedup"],
                    line_shape='spline',
                    name = point[0:50],
                    mode='lines+markers',
                    )
                ).update_layout(
                    xaxis={"title":"Function Speedup"},
                    yaxis={"title":"Program Speedup"}
                    )
    _points = list(mask_all["progress points"])
    _pointsidx = list(range(0,len(_points)))
    _count=[]
    Hist_df = px.data.tips()
    for point in _points:
        _count.append(len(list(mask_all["progress points"] == point)))
    HIST_DATA = pd.DataFrame(data= {"Function":_points})
    fig3 = px.histogram(
            HIST_DATA,
            x="Function",
            marginal="rug",
            color = "Function",
            #labels={'Function':'# of experiments'},
            height=800,
            nbins = 5
            )
    layout2  = [
            html.H4("Selected Causal Profiles", style={"color": "white"}),
        ]
    for point in sorted(list(mask_select.point.unique())):
        subplots = go.Figure()
        sub_data = mask_select[mask_select["point"] == point]
        line_number = point[point.rfind(':'):].isnumeric()
        if line_number:
            #untested
            for prog in list(sub_data["progress points"].unique()):
                sub_data_prog = sub_data[sub_data["progress points"] == prog]
                subplots.add_trace(
                    go.Scatter(
                    x = sub_data_prog["Line Speedup"],
                    y = sub_data_prog["Program Speedup"],
                    line_shape='spline',
                    name = prog,
                    mode='lines+markers',
                    )
                ).update_layout(
                    xaxis={"title":"Line Speedup"},
                    yaxis={"title":"Program Speedup"}
                    )
        else:
            for prog in list(sub_data["progress points"].unique()):
                sub_data_prog = sub_data[sub_data["progress points"] == prog]
                subplots.add_trace(
                    go.Scatter(
                    x = sub_data_prog["Line Speedup"],
                    y = sub_data_prog["Program Speedup"],
                    line_shape='spline',
                    name = prog,
                    mode='lines+markers',
                    )
                ).update_layout(
                    xaxis={"title":"Function Speedup"},
                    yaxis={"title":"Program Speedup"}
                    )
        layout2.append(html.H4(point, style={"color": "white"}))
        layout2.append(dcc.Graph(figure = subplots))
    
    layout1  = [
            html.H4("All Causal Profiles", style={"color": "white"}),
            dcc.Graph(figure = fig3)
        ]
    return mask_all, layout1, layout2


def reset_Input_filters(kernel_names, max_points):
    sortOptions = ["Alphabetical", "Max Speedup", "Min Speedup", "Impact"]

    input_filters = [
        {
            "Name": "Sort by",
            "filter": [],
            "values": list(
                map(
                    str,
                    sortOptions,
                )
            ),
            "type": "Name",
        },
        {
            "Name": "kernel",
            "filter": [],
            "values": list(
                map(
                    str,
                    kernel_names,
                )
            ),
            "type": "Kernel Name",
        },
        {
            "Name": "points",
            "filter": [],
            "values": max_points - 1,
            "type": "int",
        },
    ]
    return input_filters


def build_causal_layout(
    app,
    runs,
    input_filters_,
    path_to_dir,
    data_,
    verbose=0,
):
    """
    Build gui layout
    """
    global data
    data = data_
    global input_filters
    input_filters = input_filters_
    program_names = sorted(list(set(data.point)))

    dropDownMenuItems = [
        dbc.DropdownMenuItem("Overview", header=True),
        dbc.DropdownMenuItem(
            "All Causal Profiles",
            href="#graph_all",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Selected Causal Profiles",
            href="#graph_select",
            external_link=True,
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
                runs[path_to_dir],
                dropDownMenuItems,
                input_filters,
                filt_kernel_names,
            ),
            html.Div(id="container", children=[]),
            line_graph1,
            line_graph2
        ]
    )

    @app.callback(
        Output("container", "children"),
        Output("nav-wrap", "children"),
        Output("graph_all", "children"),
        Output("graph_select", "children"),
        [Input("nav-wrap", "children")],
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
        sort_filt,
        func_regex,
        exp_regex,
        points_filt,
        workload_path,
        list_of_contents,
        filename,
        div_children,
    ):
        global file_timestamp
        global data
        global input_filters

        CLI = False
    
        
        # change to if debug
        if True:
            print("Sort by is ", sort_filt)
            print("func_regex is ", func_regex)
            print("exp_regex is ", exp_regex)
            print("points is: ", points_filt)

        div_children = []
        files = []
        fig1 = None
        fig2 = None
        global new_data
        global func_list
        global exp_list
        func_list = sorted(list(data.point.unique()))
        exp_list = sorted(list(data["progress points"].unique()))
        
        if workload_path is not None and os.path.isdir(workload_path):
            #files = glob.glob(os.path.join(workload_path, "*.coz")) + 
            files = glob.glob(os.path.join(workload_path, "*.json"))
            #subfiles = glob.glob(os.path.join(workload_path, "*/*.coz")) + 
            subfiles = glob.glob(os.path.join(workload_path, "*/*.json"))
            metadata = glob.glob(os.path.join(workload_path, "*/metadata*.json"))

            all_files = files + subfiles
            new_data = pd.DataFrame()
            #for profile_path in all_files:
            new_data = new_data.append(parseFiles(all_files, CLI))
            new_data = new_data.rename(
                columns={"speedup": "Line Speedup", "progress_speedup": "Program Speedup"}
            )
            data = new_data

            # reset checklists
            func_list = sorted(list(data.point.unique()))
            exp_list = sorted(list(data["progress points"].unique()))

            max_points = new_data.point.value_counts().max().max()

            # reset input_filters
            input_filters = reset_Input_filters(checklist_options, max_points)

            screen_data, fig1, fig2 = update_line_graph(
                sort_filt, checklist_values, checklist_values, new_data, points_filt
            )

            header = get_header(data, dropDownMenuItems, input_filters, filt_kernel_names)
            return (
                div_children,
                header,
                fig1,
                fig2,
            )
        # div_children.append()
        elif list_of_contents is not None:
            if ".coz" in filename or ".json" in filename:
                new_data_file = base64.decodebytes(
                    list_of_contents.encode("utf-8").split(b";base64,")[1]
                ).decode("utf-8")
                # change to if debug
                
                new_data = parseUploadedFile(new_data_file, CLI)
                new_data = new_data.rename(
                    columns={
                        "speedup": "Line Speedup",
                        "progress_speedup": "Program Speedup",
                    }
                )
                data = new_data

                # reset checklists
                #checklist_options = checklist_values = sorted(list(data.point.unique()))

                max_points = new_data.point.value_counts().max().max()

                # reset input_filters
                input_filters = reset_Input_filters(checklist_options, max_points)

                screen_data, fig1, fig2 = update_line_graph(
                    sort_filt, checklist_values, checklist_values, new_data, points_filt
                )
                header = get_header(
                    data, dropDownMenuItems, input_filters, filt_kernel_names
                )
                return (
                    div_children,
                    header,
                    fig1,
                    fig2,
                )


        elif func_regex is not None or exp_regex is not None:
            # filter options and values
            if func_regex is not None:
                p = re.compile(func_regex, flags=0)

                func_list = [
                    s for s in list(data["point"].unique()) if p.match(s)
                ]
            if exp_regex is not None:
                p = re.compile(exp_regex, flags=0)

                exp_list = [
                    s for s in list(data["progress points"].unique()) if p.match(s)
                ]

            # change to update checklist after points selection
            screen_data, fig1, fig2 = update_line_graph(
                sort_filt,
                func_list,
                exp_list,
                data,
                points_filt,
            )

            # TODO keep min points value...
            return (
                div_children,
                header,
                fig1,
                fig2
            )


        else:
            # change to update checklist after points selection
            func_list = sorted(list(data.point.unique()))
            exp_list = sorted(list(data["progress points"].unique()))

            screen_data, fig1, fig2 = update_line_graph(
                sort_filt,
                func_list,
                exp_list,
                data,
                points_filt,
            )

            return (
                div_children,
                header,
                fig1,
                fig2,
            )
