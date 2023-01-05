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

import sys
import argparse
import os.path
import dash
import dash_bootstrap_components as dbc
import copy
import json
import pandas as pd

from pathlib import Path
from yaml import parse
from collections import OrderedDict

from . import gui
from .parser import parseFile
from .parser import getSpeedupData


def causal(args):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    # TODO This will become a glob to look for subfolders with coz files
    workload_path = os.path.join(args.path, "profile.coz")

    f = open(workload_path, "r")
    new_df = parseFile(workload_path)
    speedup_df = getSpeedupData(new_df).rename(
        columns={"speedup": "Line Speedup", "progress_speedup": "Program Speedup"}
    )

    runs = OrderedDict({workload_path: speedup_df})
    kernel_names = ["program1", "program2"]
    max_points = 9
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
            "values": max_points,
            "type": "int",
        },
    ]
    gui.build_causal_layout(
        app,
        runs,
        input_filters,
        workload_path,
        speedup_df,
        args.verbose,
    )
    app.run_server(debug=False, host="0.0.0.0", port=8051)


def main():
    # omnitrace version
    this_dir = Path(__file__).resolve().parent
    if os.path.basename(this_dir) == "source":
        ver_path = os.path.join(f"{this_dir.parent}", "VERSION")
    else:
        ver_path = os.path.join(f"{this_dir}", "VERSION")
    f = open(ver_path, "r")
    VER = f.read()

    my_parser = argparse.ArgumentParser(
        description="AMD's OmniTrace GUI",
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
        usage="""
                                        \nomnitrace-causal-plot --path <path>

                                        \n\n-------------------------------------------------------------------------------
                                        \nExamples:
                                        \n\tomnitrace-causal-plot --path workloads/toy
                                        \n-------------------------------------------------------------------------------\n
                                        """,
    )
    my_parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="Causal Visualizer (" + VER + ")",
    )

    my_parser.add_argument(
        "-V", "--verbose", help="Increase output verbosity", default=0, type=int,
    )

    my_parser.add_argument(
        "-p",
        "--path",
        metavar="",
        type=str,
        dest="path",
        default=os.path.join(os.path.dirname(__file__), "workloads", "toy"),
        required=False,
        help="\t\t\tSpecify path to save workload.\n\t\t\t(DEFAULT: {}/workloads/<name>)".format(
            os.getcwd()
        ),
    )

    args = my_parser.parse_args()
    causal(args)


if __name__ == "__main__":
    main()