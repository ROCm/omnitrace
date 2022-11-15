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
def fake_data():
    df = pd.DataFrame(
        [
            ["program1",0,1],["program1", 1,1],["program1", 2,1],["program1",3,1],["program1",4,1],
        ],
        columns=list(["Program","Line speedup","Program Speedup"])
        )
    df2 = pd.DataFrame(
        [
            ['program2',0,0],['program2',1,1],['program2',2,2],['program2',3,3],['program2',4,4],['program2',5,5],['program2',6,6],['program2',7,7],['program2',8,8]
        ],
        columns=list(["Program","Line speedup","Program Speedup"])
        )
    df = df.append(df2,ignore_index=True)
    return df

def causal(args):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    # TODO This will become a glob to look for subfolders with coz files
    workload_path = os.path.join(args.path, "profile.coz")

    f = open(workload_path, "r")
    df = fake_data()
    new_df=parseFile(workload_path)
    speedup_df=getSpeedupData(new_df).rename(columns={"speedup": "Line Speedup","progress_speedup": "Program Speedup" })

    runs = OrderedDict({workload_path: speedup_df})
    kernel_names = ["program1","program2"]
    max_points = 9
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
            "values": list(
                map(
                    str,
                    range(max_points),
                )
            ),
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
    app.run_server(debug=False, host="0.0.0.0", port=8050)


def miperf(args):
    avail_normalizations = ["per Wave", "per Cycle", "per Sec", "per Kernel"]
    soc_params_dir = os.path.join(os.path.dirname(__file__), "..", "soc_params")

    soc_spec_df = file_io.load_soc_params(soc_params_dir)

    # NB: maybe create bak file for the old run before open it
    output = open(args.output_file, "w+") if args.output_file else sys.stdout

    single_panel_config = file_io.is_single_panel_config(Path(args.config_dir))
    archConfigs = {}
    for arch in file_io.supported_arch.keys():
        ac = schema.ArchConfig()
        if args.list_kernels:
            ac.panel_configs = file_io.top_stats_build_in_config
        else:
            arch_panel_config = (
                args.config_dir
                if single_panel_config
                else args.config_dir.joinpath(arch)
            )
            ac.panel_configs = file_io.load_panel_configs(arch_panel_config)

        # TODO: filter_metrics should/might be one per arch
        # print(ac)

        parser.build_dfs(ac, args.filter_metrics)

        archConfigs[arch] = ac

    for k, v in archConfigs.items():
        parser.build_metric_value_string(v.dfs, v.dfs_type, args.normal_unit)

    runs = OrderedDict()
    # err checking for multiple runs and multiple gpu_kernel filter
    # TODO: move it to util
    if args.gpu_kernel and (len(args.path) != len(args.gpu_kernel)):
        if len(args.gpu_kernel) == 1:
            for i in range(len(args.path) - 1):
                args.gpu_kernel.extend(args.gpu_kernel)
        else:
            print(
                "Error: the number of --filter-kernels doesn't match the number of --dir.",
                file=output,
            )
            sys.exit(-1)

    # Todo: warning single -d with multiple dirs
    for d in args.path:
        w = schema.Workload()
        w.sys_info = file_io.load_sys_info(Path(d[0], "sysinfo.csv"))
        w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
        arch = w.sys_info.iloc[0]["gpu_soc"]
        w.dfs = copy.deepcopy(archConfigs[arch].dfs)
        w.dfs_type = archConfigs[arch].dfs_type
        w.soc_spec = file_io.get_soc_params(soc_spec_df, arch)
        runs[d[0]] = w

    # Filtering
    if args.gpu_kernel:
        for d, gk in zip(args.path, args.gpu_kernel):
            runs[d[0]].filter_kernel_ids = gk

    if args.gpu_id:
        if len(args.gpu_id) == 1 and len(args.path) != 1:
            for i in range(len(args.path) - 1):
                args.gpu_id.extend(args.gpu_id)
        for d, gi in zip(args.path, args.gpu_id):
            runs[d[0]].filter_gpu_ids = gi
    # NOTE: INVALID DISPATCH IDS ARE NOT CAUGHT HERE. THEY CAUSE AN ERROR IN TABLE GENERATION!!!!!!!!!
    if args.gpu_dispatch_id:
        if len(args.gpu_dispatch_id) == 1 and len(args.path) != 1:
            for i in range(len(args.path) - 1):
                args.gpu_dispatch_id.extend(args.gpu_dispatch_id)
        for d, gd in zip(args.path, args.gpu_dispatch_id):
            runs[d[0]].filter_dispatch_ids = gd

    if args.gui:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

        if len(runs) == 1:
            num_results = 10
            file_io.create_df_kernel_top_stats(
                args.path[0][0],
                runs[args.path[0][0]].filter_gpu_ids,
                runs[args.path[0][0]].filter_dispatch_ids,
                args.time_unit,
                num_results,
            )
            runs[args.path[0][0]].raw_pmc = file_io.create_df_pmc(
                args.path[0][0]
            )  # create mega df
            is_gui = False
            parser.load_table_data(
                runs[args.path[0][0]], args.path[0][0], is_gui, args.g
            )  # create the loaded table

            input_filters = {
                "kernel": {
                    "filter": runs[args.path[0][0]].filter_kernel_ids,
                    "values": list(
                        map(
                            str,
                            runs[args.path[0][0]].raw_pmc[
                                schema.pmc_perf_file_prefix
                            ]["Index"],
                        )
                    ),
                    "type": "Kernel Name",
                },
                "gpu": {
                    "filter": runs[args.path[0][0]].filter_gpu_ids,
                    "values": list(
                        map(
                            str,
                            runs[args.path[0][0]].raw_pmc[
                                schema.pmc_perf_file_prefix
                            ]["gpu-id"],
                        )
                    ),
                    "type": "int",
                },
                "dispatch": {
                    "filter": runs[args.path[0][0]].filter_dispatch_ids,
                    "values": list(
                        map(
                            str,
                            runs[args.path[0][0]].raw_pmc[
                                schema.pmc_perf_file_prefix
                            ]["Index"],
                        )
                    ),
                    "type": "int",
                },
                "Normalization": {
                    "filter": "per Wave",
                    "values": avail_normalizations,
                },
            }

            gui.build_miperf_layout(
                app,
                runs,
                archConfigs["gfx90a"],
                input_filters,
                args.decimal,
                args.time_unit,
                args.cols,
                str(args.path[0][0]),
                args.g,
                args.verbose,
            )
            app.run_server(debug=False, host="0.0.0.0", port=args.gui)
        else:
            print("Multiple runs not supported yet")


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
    Causal_parser.add_argument(
        "-g", action="store_true", help="\t\tDebug single metric."
    )
    Causal_parser.add_argument(
        "-V", "--verbose", help="Increase output verbosity", action="store_true"
    )

    Miperf_parser = subparsers.add_parser(
        "Miperf",
        help="Miperf's GUI",
        usage="""
                                        \nmiperf -p workloads/mixbench/mi200/

                                        \n\n-------------------------------------------------------------------------------
                                        \nExamples:
                                        \n\tmiperf -p workloads/mixbench/mi200/
                                        \n-------------------------------------------------------------------------------\n
                                        """,
        prog="tool",
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )

    args = my_parser.parse_args()
    if args.mode == "Causal":
        print("hello")
        causal(args)
    if args.mode == "Miperf":
        print("Welcome to Uncharted Territory")
        # TODO add miperf filter args to that mode.
        miperf(args)


if __name__ == "__main__":
    main()
