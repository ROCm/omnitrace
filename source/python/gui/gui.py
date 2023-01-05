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

import re
from re import L
from selectors import EpollSelector
import sys
import copy
from matplotlib.axis import XAxis
from numpy import append
import pandas as pd
import base64
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme, Symbol
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
import os
import plotly.express as px
from os.path import exists

import colorlover
from pyparsing import line_end

from source.utils import parser, file_io, schema

from source.components.header import get_header
from source.components.roofline import get_roofline
from source.components.memchart import get_memchart
from source.utils.causal_parser import parseFile, parseUploadedFile, getSpeedupData
import glob

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
        [
            html.H4("All Causal Profiles", style={"color": "white"}),
            dcc.Graph(
                id="graph_all",
                config={"responsive": True},
            ),
            dcc.Checklist(
                id="checklist_all", options=data_options, value=data_options, inline=True
            ),
        ]
    )
    layout2 = html.Div(
        [
            html.H4("Selected Causal Profiles", style={"color": "white"}),
            dcc.Graph(id="graph_select"),
            dcc.Checklist(
                id="checklist_select",
                options=data_options,
                value=data_options,
                inline=True,
            ),
        ]
    )

    return layout1, layout2


def update_line_graph(sort_filt, selected_all, selected_select, data, points_filt):

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
    # points_filt = 6.5
    sufficient_points = point_counts > points_filt
    sufficient_points = sufficient_points.loc[lambda x: x == True]
    sufficient_points = list(sufficient_points.index)
    # sufficient_points = sufficient_points
    mask_all = data[data.point.isin(selected_all)]
    mask_all = mask_all[mask_all.point.isin(sufficient_points)]

    mask_select = data[data.point.isin(selected_select)]
    mask_select = mask_select[mask_select.point.isin(sufficient_points)]
    # what = mask_select.value_counts()[True]
    # what = data[mask_all]
    # fig_data1 = data[mask_all]
    # fig_data2 = data[mask_select]
    fig1 = px.line(
        mask_all,
        x="Line Speedup",
        y="Program Speedup",
        # height=700,
        # width=700,
        # automargin="center",
        # margin={l:50, r:50},
        color="point",
        markers=True,
        line_shape="spline",
    )
    fig2 = px.line(
        mask_select,
        x="Line Speedup",
        y="Program Speedup",
        # height=700, width=700,
        color="point",
        markers=True,
        facet_col="point",
        facet_col_wrap=3,
        line_shape="spline",
    )

    return mask_all, fig1, fig2


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
    debug=False,
    verbose=False,
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
            line_graph2,
        ]
    )

    @app.callback(
        Output("container", "children"),
        Output("nav-wrap", "children"),
        Output("graph_all", "figure"),
        Output("graph_select", "figure"),
        Output("checklist_all", "options"),
        Output("checklist_select", "options"),
        Output("checklist_all", "value"),
        Output("checklist_select", "value"),
        [Input("nav-wrap", "children")],
        [Input("Sort by-filt", "value")],
        [Input("point-regex", "value")],
        [Input("points-filt", "value")],
        [Input("checklist_all", "value")],
        [Input("checklist_select", "value")],
        [Input("file-path", "value")],
        [Input("upload-drag", "contents")],
        [State("upload-drag", "filename")],
        [State("container", "children")],
    )
    def generate_from_filter(
        header,
        sort_filt,
        point_regex,
        points_filt,
        checklist_all_values,
        checklist_select_values,
        workload_path,
        list_of_contents,
        filename,
        div_children,
    ):
        global file_timestamp
        global data
        global input_filters
        global checklist_options
        global checklist_values

        # change to if debug
        if True:
            print("Sort by is ", sort_filt)
            print("point_regex is ", point_regex)
            print("points is: ", points_filt)
            print("checklist_all is: ", checklist_all_values)
            print("checklist_select is: ", checklist_select_values)

        div_children = []
        files = []
        fig1 = None
        fig2 = None
        global new_data

        if workload_path is not None and os.path.isdir(workload_path):
            files = glob.glob(os.path.join(workload_path, "*.coz"))
            subfiles = glob.glob(os.path.join(workload_path, "*/*.coz"))
            metadata = glob.glob(os.path.join(workload_path, "*/metadata*.json"))

            all_files = files + subfiles
            new_data = pd.DataFrame()
            for profile_path in all_files:
                new_data = new_data.append(parseFile(profile_path))
            new_data = getSpeedupData(new_data).rename(
                columns={"speedup": "Line Speedup", "progress_speedup": "Program Speedup"}
            )
            data = new_data

            # reset checklists
            checklist_options = checklist_values = sorted(list(data.point.unique()))

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
                checklist_options,
                checklist_options,
                checklist_options,
                checklist_options,
            )
        # div_children.append()
        elif list_of_contents is not None:
            if ".coz" in filename:
                new_data_file = base64.decodebytes(
                    list_of_contents.encode("utf-8").split(b";base64,")[1]
                ).decode("utf-8")
                new_data = parseUploadedFile(new_data_file)
                new_data = getSpeedupData(new_data).rename(
                    columns={
                        "speedup": "Line Speedup",
                        "progress_speedup": "Program Speedup",
                    }
                )
                data = new_data

                # reset checklists
                checklist_options = checklist_values = sorted(list(data.point.unique()))

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
                    checklist_options,
                    checklist_options,
                    checklist_options,
                    checklist_options,
                )

        elif point_regex is not None:
            # filter options and values
            p = re.compile(point_regex, flags=0)

            checklist_all_values = checklist_select_values = [
                s for s in checklist_values if p.match(s)
            ]
            checklist_select_options = checklist_all_options = [
                s for s in checklist_options if p.match(s)
            ]

            # change to update checklist after points selection
            screen_data, fig1, fig2 = update_line_graph(
                sort_filt,
                checklist_all_values,
                checklist_select_values,
                data,
                points_filt,
            )
            screen_data_points = sorted(list(screen_data.point.unique()))

            # TODO keep min points value...
            return (
                div_children,
                header,
                fig1,
                fig2,
                checklist_all_options,
                checklist_select_options,
                checklist_all_values,
                checklist_select_values,
            )

        else:
            # change to update checklist after points selection
            screen_data, fig1, fig2 = update_line_graph(
                sort_filt,
                checklist_all_values,
                checklist_select_values,
                data,
                points_filt,
            )
            screen_data_points = sorted(list(screen_data.point.unique()))

            # First run, checklist options not populated yet
            if checklist_options is None:
                checklist_all_values = (
                    checklist_select_values
                ) = (
                    checklist_select_options
                ) = (
                    checklist_all_options
                ) = checklist_values = checklist_options = screen_data_points

            else:
                # TODO filter checklist_options to include only screen_data_points...maybe values also
                checklist_all_values = checklist_all_values
                checklist_select_values = checklist_select_values
                checklist_select_options = checklist_all_options = checklist_options
            # TODO keep min points value..........
            return (
                div_children,
                header,
                fig1,
                fig2,
                checklist_all_options,
                checklist_select_options,
                checklist_all_values,
                checklist_select_values,
            )
