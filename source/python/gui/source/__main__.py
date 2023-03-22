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
from .parser import compute_speedups, process_data, process_samples, compute_sorts
from . import __version__


def causal(args):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    # TODO This will become a glob to look for subfolders with coz files
    workload_path = [args.path]

    num_stddev = args.stddev
    num_speedups = len(args.speedups)

    if num_speedups > 0 and args.num_points > num_speedups:
        args.num_points = num_speedups

    results_df = pd.DataFrame()
    samp = {}
    runs_dict = {}
    inp = args.path
    if os.path.exists(inp):
        if os.path.isdir(inp):
            inp = glob.glob(os.path.join(inp, "*.json"))
        elif os.path.isfile(inp):
            inp = [inp]
        for file in inp:
            with open(file, "r") as f:
                inp_data = json.load(f)
                file_name = os.path.basename(file)
                _data = process_data({}, inp_data, args.experiments, args.progress_points)
                _samp = process_samples({}, inp_data)
                runs_dict[file_name] = _data
                samp.update(_samp)

        results_df = compute_sorts(
            compute_speedups(
                runs_dict,
                args.speedups,
                args.num_points,
                args.validate,
                True if args.cli or args.verbose >= 3 else False,
            )
        )

    samples_df = pd.DataFrame(
        [{"location": loc, "count": count} for loc, count in sorted(samp.items())]
    )

    if not args.cli:
        max_points = 9
        sortOptions = ["Alphabetical", "Max Speedup", "Min Speedup", "Impact"]
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
                "values": list(runs_dict.keys()),
                "default": list(runs_dict.keys()),
                "type": "Name",
                "multi": True,
            },
            {"Name": "points", "filter": [], "values": max_points, "type": "int"},
        ]

        gui.build_causal_layout(
            app,
            runs_dict,
            input_filters,
            args.path,
            results_df,
            samples_df,
            args.verbose,
            args.light,
        )
        app.run_server(
            debug=True if args.verbose >= 3 else False,
            host=args.ip_address,
            port=args.ip_port,
        )


def main():
    settings = {}

    this_dir = Path(__file__).resolve().parent
    if os.path.basename(this_dir) == "source":
        settings_path = os.path.join(f"{this_dir.parent}", "settings.json")
    else:
        settings_path = os.path.join(f"{this_dir}", "settings.json")
    if os.path.exists(settings_path):
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
        "--version",
        action="version",
        version="OmniTrace Causal Viewer v{}\n".format(f"{__version__}".strip("\n")),
    )

    my_parser.add_argument(
        "-c",
        "--cli",
        action="store_true",
        required=False,
        default=settings["cli"] if "cli" in settings else False,
        help="Do not launch the GUI, print the causal analysis out to the console only",
    )

    my_parser.add_argument(
        "-l",
        "--light",
        action="store_true",
        required=False,
        default=settings["light"] if "light" in settings else False,
        help="light Mode",
    )

    my_parser.add_argument(
        "-w",
        "--workload",
        metavar="FOLDER",
        type=str,
        dest="path",
        default=settings["path"] if "path" in settings else "",
        required=False,
        help="Specify path to causal profiles.\n(DEFAULT: {}/workloads/<name>)".format(
            os.getcwd()
        ),
    )

    my_parser.add_argument(
        "-V",
        "--verbose",
        help="Increase output verbosity, if 3 or greater, CLI output will show in terminal in GUI mode",
        default=0,
        type=int,
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

    my_parser.add_argument(
        "-e", "--experiments", type=str, help="Regex for experiments", default=".*"
    )

    my_parser.add_argument(
        "-p",
        "--progress-points",
        type=str,
        help="Regex for progress points",
        default=".*",
    )

    my_parser.add_argument(
        "-n", "--num-points", type=int, help="Minimum number of data points", default=5
    )

    my_parser.add_argument(
        "-s",
        "--speedups",
        type=int,
        help="List of speedup values to report",
        nargs="*",
        default=[],
    )

    my_parser.add_argument(
        "-d",
        "--stddev",
        type=int,
        help="Number of standard deviations to report",
        default=1,
    )

    my_parser.add_argument(
        "-v",
        "--validate",
        type=str,
        nargs="*",
        help="Validate speedup: {experiment regex} {progress-point regex} {virtual-speedup} {expected-speedup} {tolerance}",
        default=[],
    )

    args = my_parser.parse_args()

    causal(args)


if __name__ == "__main__":
    main()
