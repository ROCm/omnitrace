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

file_timestamp = 0
data=pd.DataFrame()
input_filters=None
checklist_options=None
checklist_values=None
pd.set_option(
    "mode.chained_assignment", None
)  # ignore SettingWithCopyWarning pandas warning

HIDDEN_SECTIONS = ["Memory Chart Analysis", "Kernels"]
HIDDEN_COLUMNS = ["Tips", "coll_level"]
IS_DARK = True  # default dark theme

# Add any elements you'd like displayed as a bar chart
barchart_elements = [
    1001,  # Instr mix
    1002,  # VALU Arith Instr mix
    1101,  # Compute pipe SOL
    1201,  # LDS SOL
    1301,  # Instruc cache SOL
    1401,  # SL1D cache SOL
    1601,  # VL1D cache SOL
    1701,  # L2 cache SOL
]


def filter_df(column, df, filt):
    filt_df = df
    if filt != []:
        filt_df = df.loc[
            df[schema.pmc_perf_file_prefix][column].astype(str).isin(filt)
        ]
    return filt_df


def discrete_background_color_bins(df, n_bins=5, columns="all"):

    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == "all":
        if "id" in df:
            df_numeric_columns = df.select_dtypes("number").drop(["id"], axis=1)
        else:
            df_numeric_columns = df.select_dtypes("number")
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [((df_max - df_min) * i) + df_min for i in bounds]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]["seq"]["Blues"][i - 1]
        color = "white" if i > len(bounds) / 2.0 else "inherit"

        for column in df_numeric_columns:
            styles.append(
                {
                    "if": {
                        "filter_query": (
                            "{{{column}}} >= {min_bound}"
                            + (
                                " && {{{column}}} < {max_bound}"
                                if (i < len(bounds) - 1)
                                else ""
                            )
                        ).format(
                            column=column,
                            min_bound=min_bound,
                            max_bound=max_bound,
                        ),
                        "column_id": column,
                    },
                    "backgroundColor": backgroundColor,
                    "color": color,
                }
            )
        legend.append(
            html.Div(
                style={"display": "inline-block", "width": "60px"},
                children=[
                    html.Div(
                        style={
                            "backgroundColor": backgroundColor,
                            "borderLeft": "1px rgb(50, 50, 50) solid",
                            "height": "10px",
                        }
                    ),
                    html.Small(
                        round(min_bound, 2), style={"paddingLeft": "2px"}
                    ),
                ],
            )
        )

    return (styles, html.Div(legend, style={"padding": "5px 0 5px 0"}))


def build_bar_chart(display_df, table_config):
    d_figs = []

    # Insr Mix bar chart
    if table_config["id"] == 1001 or table_config["id"] == 1002:
        display_df["Count"] = [
            x.astype(int) if x != "" else int(0) for x in display_df["Count"]
        ]
        df_unit = display_df["Unit"][0]
        d_figs.append(
            px.bar(
                display_df,
                x="Count",
                y="Metric",
                color="Count",
                labels={"Count": "# of {}".format(df_unit.lower())},
                height=400,
                orientation="h",
            )
        )

    # Speed-of-light bar chart
    else:
        display_df["Value"] = [
            x.astype(float) if x != "" else float(0)
            for x in display_df["Value"]
        ]
        if table_config["id"] == 1701:
            # special layout for L2 Cache SOL
            d_figs.append(
                px.bar(
                    display_df[display_df["Unit"] == "Pct"],
                    x="Value",
                    y="Metric",
                    color="Value",
                    range_color=[0, 100],
                    labels={"Value": "%"},
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, 110], ticks="inside")
            )  # append first % chart
            d_figs.append(
                px.bar(
                    display_df[display_df["Unit"] == "Gb/s"],
                    x="Value",
                    y="Metric",
                    color="Value",
                    range_color=[0, 1638],
                    labels={"Value": "GB/s"},
                    height=220,
                    orientation="h",
                ).update_xaxes(range=[0, 1638])
            )  # append second GB/s chart
        else:
            d_figs.append(
                px.bar(
                    display_df,
                    x="Value",
                    y="Metric",
                    color="Value",
                    range_color=[0, 100],
                    labels={"Value": "%"},
                    height=400,
                    orientation="h",
                ).update_xaxes(range=[0, 110])
            )

    # update layout for each of the charts
    for fig in d_figs:
        fig.update_layout(
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
        )
    return d_figs


def build_table_chart(
    display_df,
    table_config,
    original_df,
    display_columns,
    comparable_columns,
    decimal,
):
    d_figs = []

    # build comlumns/header with formatting
    formatted_columns = []
    for col in display_df.columns:
        if (
            str(col).lower() == "pct"
            or str(col).lower() == "pop"
            or str(col).lower() == "percentage"
        ):
            formatted_columns.append(
                dict(
                    id=col,
                    name=col,
                    type="numeric",
                    format={"specifier": ".{}f".format(decimal)},
                )
            )
        elif col in comparable_columns:
            formatted_columns.append(
                dict(
                    id=col,
                    name=col,
                    type="numeric",
                    format={"specifier": ".{}f".format(decimal)},
                )
            )
        else:
            formatted_columns.append(dict(id=col, name=col, type="text"))

    # tooltip shows only on the 1st col for now if 'Tips' available
    table_tooltip = (
        [
            {
                column: {
                    "value": str(row["Tips"])
                    if column == display_columns[0] and row["Tips"]
                    else "",
                    "type": "markdown",
                }
                for column, value in row.items()
            }
            for row in original_df.to_dict("records")
        ]
        if "Tips" in original_df.columns.values.tolist()
        else None
    )

    # build data table with columns, tooltip, df and other properties
    d_t = dash_table.DataTable(
        id=str(table_config["id"]),
        sort_action="native",
        sort_mode="multi",
        columns=formatted_columns,
        tooltip_data=table_tooltip,
        # left-aligning the text of the 1st col
        style_cell_conditional=[
            {"if": {"column_id": display_columns[0]}, "textAlign": "left"}
        ],
        # display style
        style_header={
            "backgroundColor": "rgb(30, 30, 30)",
            "color": "white",
            "fontWeight": "bold",
        }
        if IS_DARK
        else {},
        style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"}
        if IS_DARK
        else {},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(60, 60, 60)"}
        ]
        if IS_DARK
        else [],
        # the df to display
        data=display_df.to_dict("records"),
    )
    # print("DATA: \n", display_df.to_dict('records'))
    d_figs.append(d_t)
    return d_figs
    # print(d_t.columns)

def build_line_graph(data, KernelName, points_filt):
    #df = px.data.gapminder() # replace with your own data source
    #mask = df.continent.isin(continents)
    data_options = sorted(list(set(data.point)))
    layout1 =    html.Div([
        html.H4('All Causal Profiles',style={'color': "white"}),
        dcc.Graph(id="graph_all",
            config={'responsive': True},
            
            ),
        dcc.Checklist(
            id="checklist_all",
            options=data_options,
            value=data_options,
            inline=True
        ),
    ])
    layout2 =    html.Div([
        html.H4('Selected Causal Profiles',style={'color': "white"}),
        dcc.Graph(id="graph_select"),
        dcc.Checklist(
            id="checklist_select",
            options=data_options,
            value=data_options,
            inline=True
        ),
    ])

    return layout1, layout2

def update_line_graph(sort_filt, selected_all, selected_select, data, points_filt):

    #df = px.data.gapminder() # replace with your own data source
    if "Alphabetical" in sort_filt:
        data  = data.sort_values(by="point")
    if "Impact" in sort_filt:
        newData=pd.DataFrame()
        impactOrder = pd.DataFrame(data.point.unique(), columns=["Program"])
        
        for index_imp, curr in impactOrder.iterrows():
            prev=pd.Series()
            data_subset = data[data["point"]== curr.Program]
            area=0
            norm_area = 0
            for index_sub, data_point in data_subset.iterrows():
                if prev.empty:
                    prev = data_point
                else:
                    avg_progress_speedup = (prev["Program Speedup"] + data_point["Program Speedup"]) / 2
                    area = area + avg_progress_speedup * (data_point["Line Speedup"] - prev["Line Speedup"])
                    norm_area = area / data_point["Line Speedup"]
                    prev = data_point
            impactOrder.at[index_imp, "area"] = norm_area
        impactOrder = impactOrder.sort_values(by="area")
        impactOrder = impactOrder.Program.unique()
    
        #add to newData in impact order
        for point in impactOrder:
            data_subset = data[data["point"] == point]
            newData = pd.concat([
                data_subset,
                newData
            ])
        data = newData
    if "Max Speedup" in sort_filt:
        speedupOrder = data.sort_values(by="Program Speedup").point.unique()
        newData=pd.DataFrame()
        for point in speedupOrder:
            data_subset = data[data["point"] == point]
            newData = pd.concat([
                data_subset,
                newData
            ])
        data = newData
    if "Min Speedup" in sort_filt:
        speedupOrder = data.sort_values(by="Program Speedup", ascending=False).point.unique()
        newData=pd.DataFrame()
        for point in speedupOrder:
            data_subset = data[data["point"] == point]
            newData = pd.concat([
                data_subset,
                newData
            ])
        data = newData

    point_counts = data.point.value_counts()
    #points_filt = 6.5
    sufficient_points = point_counts>points_filt
    sufficient_points = sufficient_points.loc[lambda x:x == True]
    sufficient_points = list(sufficient_points.index)
    #sufficient_points = sufficient_points
    mask_all = data[data.point.isin(selected_all)]
    mask_all = mask_all[mask_all.point.isin(sufficient_points)]

    mask_select = data[data.point.isin(selected_select)]
    mask_select = mask_select[mask_select.point.isin(sufficient_points)]
    #what = mask_select.value_counts()[True]
    #what = data[mask_all]
    #fig_data1 = data[mask_all]
    #fig_data2 = data[mask_select]
    fig1 = px.line(mask_all, 
        x="Line Speedup", 
        y="Program Speedup", 
        #height=700, 
        #width=700, 
        #automargin="center",
        #margin={l:50, r:50},
        color="point",
        markers=True,
        line_shape="spline"
        )
    fig2 = px.line(mask_select, 
        x="Line Speedup", y="Program Speedup", 
        #height=700, width=700, 
        color="point",
        markers=True,
        facet_col="point",
        facet_col_wrap = 3,
        line_shape="spline"
        )

    return mask_all, fig1, fig2

def reset_Input_filters(kernel_names, max_points):
    sortOptions=[
        "Alphabetical",
        "Max Speedup",
        "Min Speedup",
        "Impact"
    ]

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
            "type": "Name",
        },
        {
            "Name": "points",
            "filter": [],
            "values": max_points-1,
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
    data =data_
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
        )
    ]

    

    inital_min_points = 3

    app.layout = html.Div(
        style={"backgroundColor": "rgb(50, 50, 50)" if IS_DARK else ""}
    )

    filt_kernel_names = []
    line_graph1,line_graph2 = build_line_graph(data, filt_kernel_names, inital_min_points)
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
        Output("graph_all", "figure"),
        Output("graph_select", "figure"),
        
        Output("checklist_all", "options"),
        Output("checklist_select", "options"),
        Output("checklist_all", "value"),
        Output("checklist_select", "value"),
        

        [Input("Sort by-filt", "value")],
        [Input("kernel-filt", "value")],
        [Input("points-filt", "value")],
        [Input("checklist_all", "value")],
        [Input("checklist_select", "value")],

        [Input("file-path", "value")],

        [Input("upload-drag", "contents")],

        [State("upload-drag", "filename")],
        #[State('upload-drag', 'last_modified')],
        #[State("checklist_all", "options")],
        #[State("checklist_select", "options")],
        [State("container", "children")],
        
    )
    def generate_from_filter(
    sort_filt, kernel_filt, points_filt, checklist_all_values, checklist_select_values,
    workload_path,
    list_of_contents,
    filename,
    #checklist_all_options, checklist_select_options, 
    div_children,
    ):
        global file_timestamp
        global data
        global input_filters
        global checklist_options
        global checklist_values

        #change to if debug
        if True:
            print("Sort by is ", sort_filt)
            print("kernel-filter is ", kernel_filt)
            print("points is: ", points_filt)
            print("checklist_all is: ", checklist_all_values)
            print("checklist_select is: ", checklist_select_values)

        div_children = []
        files = []
        fig1=None
        fig2=None
        global new_data

        if workload_path is not None and exists(os.path.join(workload_path, "profile.coz")):
            profile_path = os.path.join(workload_path, "profile.coz")
            #if exists(os.path.join(workload_path, "profile.coz")):
            new_data = parseFile(profile_path)
            new_data = getSpeedupData(new_data).rename(columns={"speedup": "Line Speedup","progress_speedup": "Program Speedup" })
            data = new_data

            #reset checklists 
            checklist_options = checklist_values = sorted(list(new_data.point.unique()))

            #reset input_filters

            screen_data, fig1, fig2 = update_line_graph(sort_filt, checklist_values, checklist_values, new_data, points_filt)

            header = get_header(data, dropDownMenuItems, input_filters, filt_kernel_names)
            return div_children, header, fig1, fig2, checklist_all_options, checklist_select_options, checklist_all_values, checklist_select_values
        #div_children.append()
        elif list_of_contents is not None:
        #if filename is not None and upload is not None:
            if ".coz" in filename:
                #file_timestamp = last_modified
                #for name, file_data in zip(filename, list_of_contents):
                new_data_file = base64.decodebytes(list_of_contents.encode("utf-8").split(b";base64,")[1]).decode("utf-8") 
                new_data = parseUploadedFile(new_data_file)
                new_data = getSpeedupData(new_data).rename(columns={"speedup": "Line Speedup","progress_speedup": "Program Speedup" })
                data = new_data

                #reset checklists
                checklist_options = checklist_values = sorted(list(new_data.point.unique()))

                max_points= new_data.point.value_counts().max().max()

                #reset input_filters
                input_filters=reset_Input_filters(checklist_options, max_points)

                screen_data, fig1, fig2 = update_line_graph(sort_filt, checklist_values, checklist_values, new_data, points_filt)
                header = get_header(data, dropDownMenuItems, input_filters, filt_kernel_names)
                return div_children, header, fig1, fig2, checklist_all_options, checklist_select_options, checklist_all_values, checklist_select_values
            
        else:
            #change to update checklist after points selection
            screen_data, fig1, fig2 = update_line_graph(sort_filt, checklist_all_values, checklist_select_values, data, points_filt)
            screen_data_points = sorted(list(screen_data.point.unique()))

            #First run, checklist options not populated yet
            if checklist_options is None:
                checklist_all_values = checklist_select_values = checklist_select_options = checklist_all_options = checklist_options = screen_data_points

            else:
                #TODO filter checklist_options to include only screen_data_points...maybe values also
                checklist_all_values = checklist_all_values
                checklist_select_values = checklist_select_values
                checklist_select_options = checklist_all_options = checklist_options

            header = get_header(data, dropDownMenuItems, input_filters, filt_kernel_names)
            return div_children, header, fig1, fig2, checklist_all_options, checklist_select_options, checklist_all_values, checklist_select_values
    

def build_miperf_layout(
    app,
    runs,
    archConfigs,
    input_filters,
    decimal,
    time_unit,
    cols,
    path_to_dir,
    debug=False,
    verbose=False,
):
    """
    Build gui layout
    """
    dropDownMenuItems = [
        dbc.DropdownMenuItem("Overview", header=True),
        dbc.DropdownMenuItem(
            "Roofline",
            href="#roofline",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Top Stat",
            href="#top_stat",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "System Info",
            href="#system_info",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "System Speed-of-Light",
            href="#system_speed-of-light",
            external_link=True,
        ),
        dbc.DropdownMenuItem("Compute", header=True),
        dbc.DropdownMenuItem(
            "Command Processor (CPF/CPC)",
            href="#command_processor_cpccpf",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Shader Processor Input (SPI)",
            href="#shader_processor_input_spi",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Wavefront",
            href="#wavefront",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Compute Units - Instruction Mix",
            href="#compute_units_-_instruction_mix",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Compute Units - Compute Pipeline",
            href="#compute_units_-_compute_pipeline",
            external_link=True,
        ),
        dbc.DropdownMenuItem("Cache", header=True),
        dbc.DropdownMenuItem(
            "Local Data Share (LDS)",
            href="#local_data_sharelds",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Instruction Cache",
            href="#instruction_cache",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Scalar L1 Data Cache",
            href="#scalar_l1_data_cache",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Texture Addresser & Texture Data (TA/TD)",
            href="#texture_addresser_and_texture_data_tatd",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "Vector L1 Data Cache",
            href="#vector_l1_data_cache",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "L2 Cache",
            href="#l2_cache",
            external_link=True,
        ),
        dbc.DropdownMenuItem(
            "L2 Cache (per channel)",
            href="#l2_cache_per_channel",
            external_link=True,
        ),
    ]
    comparable_columns = parser.build_comparable_columns(time_unit)

    base_run, base_data = next(iter(runs.items()))

    app.layout = html.Div(
        style={"backgroundColor": "rgb(50, 50, 50)" if IS_DARK else ""}
    )

    filt_kernel_names = []
    kernel_top_df = runs[path_to_dir].dfs[1]
    for kernel_id in runs[path_to_dir].filter_kernel_ids:
        filt_kernel_names.append(kernel_top_df.loc[kernel_id, "KernelName"])

    app.layout.children = html.Div(
        children=[
            get_header(
                runs[path_to_dir].raw_pmc,
                dropDownMenuItems,
                input_filters,
                filt_kernel_names,
            ),
            html.Div(id="container", children=[]),
        ]
    )

    @app.callback(
        Output("container", "children"),
        [Input("disp-filt", "value")],
        [Input("kernel-filt", "value")],
        [Input("gcd-filt", "value")],
        [State("container", "children")],
    )
    def generate_from_filter(
        disp_filt, kernel_filter, gcd_filter, div_children
    ):
        runs[path_to_dir].dfs = copy.deepcopy(
            archConfigs.dfs
        )  # reset the equations
        # Generate original raw df
        runs[path_to_dir].raw_pmc = file_io.create_df_pmc(path_to_dir)
        if verbose:
            print("disp-filter is ", disp_filt)
            print("kernel-filter is ", kernel_filter)
            print("gpu-filter is ", gcd_filter)
        runs[path_to_dir].filter_kernel_ids = kernel_filter
        runs[path_to_dir].filter_gpu_ids = gcd_filter
        runs[path_to_dir].filter_dispatch_ids = disp_filt
        # Reload the pmc_kernel_top.csv for Top Stats panel
        num_results = 10
        file_io.create_df_kernel_top_stats(
            path_to_dir,
            runs[path_to_dir].filter_gpu_ids,
            runs[path_to_dir].filter_dispatch_ids,
            time_unit,
            num_results,
        )
        # Evaluate metrics and table data from the raw df
        is_gui = True
        parser.load_table_data(
            runs[path_to_dir], path_to_dir, True, debug
        )  # Note: All the filtering happens in this function
        div_children = []
        div_children.append(
            html.Div(
                className="row banner",
                children=[
                    html.H3(
                        children=[
                            "Placeholder. Guided Analysis coming soon..."
                        ],
                        style={"color": "white"},
                    ),
                ],
            )
        )
        div_children.append(
            html.P(
                className="scrolldown",
                children=[
                    html.A(
                        className="smoothscroll",
                        href="#roofline",
                        children=[html.I(className="icon-down-circle")],
                    )
                ],
            )
        )
        div_children.append(
            get_memchart(
                archConfigs.panel_configs[1900]["data source"], base_data
            )
        )
        # append roofline section
        div_children.append(
            get_roofline(
                path_to_dir,
                parser.apply_filters(runs[path_to_dir], is_gui, debug),
                verbose,
            )
        )
        # Iterate over each section as defined in panel configs
        for panel_id, panel in archConfigs.panel_configs.items():
            title = str(panel_id // 100) + ". " + panel["title"]
            section_title = (
                panel["title"]
                .replace("(", "")
                .replace(")", "")
                .replace("/", "")
                .replace(" ", "_")
                .lower()
            )
            html_section = []

            if panel["title"] not in HIDDEN_SECTIONS:
                # Iterate over each table per section
                for data_source in panel["data source"]:
                    for t_type, table_config in data_source.items():
                        content = []
                        original_df = base_data.dfs[table_config["id"]]

                        # The sys info table need to add index back
                        if (
                            t_type == "raw_csv_table"
                            and "Info" in original_df.keys()
                        ):
                            original_df.reset_index(inplace=True)

                        display_columns = (
                            original_df.columns.values.tolist().copy()
                        )
                        # Remove hidden columns. Better way to do it?
                        for col in HIDDEN_COLUMNS:
                            if col in display_columns:
                                display_columns.remove(col)
                        display_df = original_df[display_columns]

                        # Determine chart type:
                        # a) Barchart
                        if table_config["id"] in barchart_elements:
                            d_figs = build_bar_chart(display_df, table_config)
                            for fig in d_figs:
                                content.append(
                                    dcc.Graph(
                                        figure=fig, style={"margin": "2%"}
                                    )
                                )
                        # B) Tablechart
                        else:
                            d_figs = build_table_chart(
                                display_df,
                                table_config,
                                original_df,
                                display_columns,
                                comparable_columns,
                                decimal,
                            )
                            for fig in d_figs:
                                content.append(
                                    html.Div([fig], style={"margin": "2%"})
                                )

                        # subtitle for each table in a panel if existing
                        if "title" in table_config and table_config["title"]:
                            subtitle = (
                                str(table_config["id"] // 100)
                                + "."
                                + str(table_config["id"] % 100)
                                + " "
                                + table_config["title"]
                                + "\n"
                            )

                            content.insert(
                                0,
                                html.H4(
                                    children=subtitle,
                                    style={"color": "white" if IS_DARK else ""},
                                ),
                            )

                        # Update content for this section
                        html_section.append(
                            html.Div(className="float-child", children=content)
                        )

                # Append the new section with all of it's contents
                div_children.append(
                    html.Section(
                        id=section_title,
                        children=[
                            html.H3(
                                children=title,
                                style={"color": "white" if IS_DARK else ""},
                            ),
                            html.Div(
                                className="float-container",
                                children=html_section,
                            ),
                        ],
                    )
                )

        return div_children


