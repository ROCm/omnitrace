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

import sys
import dash_daq as daq
import dash_bootstrap_components as dbc

from dash import html, dash_table, dcc
from matplotlib.style import available

# List all the unique column values for desired column in df, 'target_col'
def list_unique(orig_list, is_numeric):
    list_set = set(orig_list)
    unique_list = list(list_set)
    if is_numeric:
        unique_list.sort()
    return unique_list


def filePath():
    return html.Div(
        className="filter",
        children=[
            html.Li(
                dcc.Input(
                    id="file-path",
                    placeholder="Insert Workload directory",
                    type="text",
                    debounce=True,
                    style={
                        "width": "100%",
                        "height": "40%",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        #'borderStyle': 'dashed',
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )
            )
        ],
    )


def function_filter(_id, _placeholder):
    return html.Li(
        className="filter",
        children=[
            html.Li(
                dcc.Input(
                    id=_id,
                    placeholder=_placeholder,
                    type="text",
                    debounce=True,
                    style={
                        "width": "100%",
                        "height": "40%",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        #'borderStyle': 'dashed',
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )
            )
        ],
    )


def uploadFile():
    return html.Div(
        className="filter",
        children=[
            html.Li(
                children=[
                    # drag and drop
                    dcc.Upload(
                        id="upload-drag",
                        children=["Drag and Drop or ", html.A("Select a File")],
                        style={
                            "width": "100%",
                            "height": "40%",
                            "lineHeight": "40%",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                        },
                    )
                ]
            )
        ],
    )


def minPoints(name, values):
    return html.Li(
        className="filter",
        # style={#'width': '100%',
        #'height': '50px'},
        children=[
            html.Div(
                style={
                    "width": "200px",
                    "position": "relative",
                    "display": "inline-block",
                    "list-style": "none",
                },
                children=[
                    html.A(
                        className="smoothscroll",
                        children=["Min Points:"],
                    ),
                ],
            ),
            html.Div(
                style={
                    "width": "200px",
                    "padding-top": "10px",
                    "vertical-align": "middle",
                    "position": "relative",
                    "display": "inline-block",
                    "list-style": "none",
                },
                children=[
                    daq.Slider(
                        min=0,
                        max=values,
                        step=1,
                        value=1,
                        id="points-filt",
                        handleLabel={"showCurrentValue": True, "label": "VALUE"},
                        size=200,
                    )
                ],
            ),
        ],
    )


def sortBy(name, values, filter, style_):
    return html.Li(
        className="filter",
        children=[
            html.Div(
                children=[
                    html.A(
                        className="smoothscroll",
                        children=[name + ":"],
                    ),
                    dcc.Dropdown(
                        list_unique(
                            values,
                            True,
                        ),
                        id=name + "-filt",
                        multi=True,
                        value=filter,
                        placeholder="ALL",
                        clearable=False,
                        style=style_,
                    ),
                ]
            )
        ],
    )


def reportBug():
    return html.Div(
        className="nav-right",
        children=[
            html.Li(
                children=[
                    # Report bug button
                    html.A(
                        href="",
                        children=[
                            html.Button(
                                className="report",
                                children=["Report Bug"],
                            )
                        ],
                    )
                ]
            )
        ],
    )


def get_header(raw_pmc, dropDownMenuItems, input_filters, kernel_names):
    children_ = [
        html.Nav(
            id="nav-wrap",
            children=[
                html.Ul(
                    id="nav",
                    children=[
                        html.Div(
                            className="nav-left",
                            children=[
                                dbc.DropdownMenu(
                                    dropDownMenuItems,
                                    label="Menu",
                                    menu_variant="dark",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ]

    for filter in input_filters:
        header_nav = children_[0].children[0].children
        if filter["type"] == "int":
            header_nav.append(minPoints(filter["Name"], filter["values"]))
        elif filter["type"] == "Name":
            header_nav.append(
                sortBy(
                    filter["Name"],
                    filter["values"],
                    filter["filter"],
                    {
                        "width": "200px",  # TODO: Change these widths to % rather than fixed value
                        "height": "34px",
                    },
                )
            )
        #elif filter["type"] == "Function Name":
        #    header_nav.append(
        #        function_filter(
        #            filter["Name"],
        #            filter["values"],
        #            filter["filter"],
        #            {
        #                "width": "200px",  # TODO: Change these widths to % rather than fixed value
        #                "height": "34px",
        #            },
        #        )
        #id    )
        else:
            print("type not supported")
            # sys.exit(1)
    header_nav = children_[0].children[0].children
    header_nav.append(function_filter("function_regex", "Funtion/line regex"))
    header_nav.append(function_filter("exp_regex", "Experiment regex"))
    header_nav.append(reportBug())
    # header_nav.append(minPoints())

    header_nav.append(filePath())
    header_nav.append(uploadFile())

    return html.Header(id="home", children=children_)
