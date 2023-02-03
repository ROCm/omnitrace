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
import glob
import pandas as pd

from pathlib import Path
from yaml import parse
from collections import OrderedDict

from . import gui
from .parser import parseFiles
from .parser import getSpeedupData


def causal(args):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    # TODO This will become a glob to look for subfolders with coz files
    workload_path = glob.glob(os.path.join(args.path, "*"), recursive=True)
    # workload_path = [os.path.join(args.path, "experiments.coz")]

    CLI = args.cli
    speedup_df = parseFiles(workload_path, CLI)
    workload_path = workload_path[0]

    if not CLI:
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
        app.run_server(
            debug=True if args.verbose >= 3 else False,
            host=args.ip_address,
            port=args.ip_port,
        )


def main():
    # omnitrace version
    this_dir = Path(__file__).resolve().parent
    if os.path.basename(this_dir) == "source":
        ver_path = os.path.join(f"{this_dir.parent}", "VERSION")
    else:
        ver_path = os.path.join(f"{this_dir}", "VERSION")
    f = open(ver_path, "r")
    VER = f.read()

    settings = {}
    if os.path.basename(this_dir) == "source":
        settings_path = os.path.join(f"{this_dir.parent}", "settings.json")
    else:
        settings_path = os.path.join(f"{this_dir}", "settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

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
        "-V",
        "--verbose",
        help="Increase output verbosity",
        default=0,
        type=int,
    )

    my_parser.add_argument(
        "-p",
        "--path",
        metavar="FOLDER",
        type=str,
        dest="path",
        default=settings["path"]
        if "path" in settings
        else os.path.join(os.path.dirname(__file__), "workloads", "toy"),
        required=False,
        help="Specify path to causal profiles.\n(DEFAULT: {}/workloads/<name>)".format(
            os.getcwd()
        ),
    )

    my_parser.add_argument(
        "--ip",
        "--ip-addr",
        metavar="IP_ADDR",
        type=str,
        dest="ip_address",
        default="0.0.0.0",
        help="Specify the IP address for the web app.\n(DEFAULT: 0.0.0.0)",
    )

    my_parser.add_argument(
        "--port",
        "--ip-port",
        metavar="PORT",
        type=int,
        dest="ip_port",
        default=8051,
        help="Specify the port number for the IP address for the web app.\n(DEFAULT: 8051)",
    )

    # only CLI
    my_parser.add_argument(
        "-c",
        "--cli",
        action="store_true",
        default=settings["cli"],
        required=False,
    )

    args = my_parser.parse_args()
    causal(args)


if __name__ == "__main__":
    main()
