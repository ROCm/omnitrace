import argparse
from mailbox import linesep
import os.path
from pathlib import Path
import sys

from yaml import parse
import gui
from source.utils import file_io, schema, parser
import dash
import dash_bootstrap_components as dbc
from collections import OrderedDict
import copy
import json
import pandas as pd
from source.utils.causal_parser import parseFile, getSpeedupData


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
        args.g,
        args.verbose,
    )
    app.run_server(debug=False, host="0.0.0.0", port=8051)


def main():
    # omnitrace version
    ver_path = cur_root = Path(__file__).resolve().parent / "VERSION"
    f = open(ver_path, "r")
    VER = f.read()
    my_parser = argparse.ArgumentParser(
        description="CLI AMD's Omnitrace GUI",
        prog="tool",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
        usage="gui --path ",
    )
    Causal_group = my_parser.add_argument_group("Causal Visualization")
    Causal_group.add_argument(
        "-v",
        "--version",
        action="version",
        version="Causal Visualizer (" + VER + ")",
    )
    subparsers = my_parser.add_subparsers(
        dest="mode",
        help="Select mode of interaction with the target application:",
    )
    Causal_parser = subparsers.add_parser(
        "Causal",
        help="Omnitrace's Causal Profiler GUI",
        usage="""
                                        \nOmnitrace gui --name <workload_name> --path <path>

                                        \n\n-------------------------------------------------------------------------------
                                        \nExamples:
                                        \n\tOmnitrace gui --name toy --path workloads/toy
                                        \n-------------------------------------------------------------------------------\n
                                        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )
    Causal_parser._optionals.title = "Help"

    Causal_parser.add_argument(
        "-p",
        "--path",
        metavar="",
        type=str,
        dest="path",
        default=os.getcwd() + "/workloads",
        required=False,
        help="\t\t\tSpecify path to save workload.\n\t\t\t(DEFAULT: {}/workloads/<name>)".format(
            os.getcwd()
        ),
    )
    Causal_parser.add_argument("-g", action="store_true", help="\t\tDebug single metric.")
    Causal_parser.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="store_true"
    )

    args = my_parser.parse_args()
    if args.mode == "Causal":
        print("hello")
        causal(args)


if __name__ == "__main__":
    main()
