#!/usr/bin/env python3

import sys
import os
import argparse
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig


def load_trace(inp, max_tries=5, retry_wait=1, bin_path=None):
    """Occasionally connecting to the trace processor fails with HTTP errors
    so this function tries to reduce spurious test failures"""

    n = 0
    tp = None

    # Check if bin_path is set and if it exists
    print("trace_processor path: ", bin_path)
    if bin_path and not os.path.isfile(bin_path):
        print(f"Path {bin_path} does not exist. Using the default path.")
        bin_path = None

    while tp is None:
        try:
            if bin_path:
                config = TraceProcessorConfig(bin_path=bin_path)
                tp = TraceProcessor(trace=inp, config=config)
            else:
                tp = TraceProcessor(trace=inp)
            break
        except Exception as ex:
            sys.stderr.write(f"{ex}\n")
            sys.stderr.flush()

            if n >= max_tries:
                raise
            else:
                import time

                time.sleep(retry_wait)
        finally:
            n += 1
    return tp


def validate_perfetto(data, labels, counts, depths):
    expected = []
    for litr, citr, ditr in zip(labels, counts, depths):
        entry = []
        _label = litr
        if ditr > 0:
            _label = "{}".format(litr)
        entry = [_label, citr, ditr]
        expected.append(entry)

    for ditr, eitr in zip(data, expected):
        _label = ditr["label"]
        _count = ditr["count"]
        _depth = ditr["depth"]

        if _label != eitr[0]:
            raise RuntimeError(f"Mismatched prefix: {_label} vs. {eitr[0]}")
        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth: {_depth} vs. {eitr[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--labels", nargs="+", type=str, help="Expected labels", default=[]
    )
    parser.add_argument(
        "-c", "--counts", nargs="+", type=int, help="Expected counts", default=[]
    )
    parser.add_argument(
        "-d", "--depths", nargs="+", type=int, help="Expected depths", default=[]
    )
    parser.add_argument(
        "-m", "--categories", nargs="+", help="Perfetto categories", default=[]
    )
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print the processed perfetto data"
    )
    parser.add_argument("-i", "--input", type=str, help="Input file", required=True)
    parser.add_argument(
        "-t", "--trace_processor_shell", type=str, help="Path of trace_processor_shell"
    )
    parser.add_argument(
        "--key-names",
        type=str,
        help="Require debug args contain a specific key",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--key-counts",
        type=int,
        help="Required number of debug args",
        default=[],
        nargs="*",
    )

    args = parser.parse_args()

    if len(args.labels) != len(args.counts) or len(args.labels) != len(args.depths):
        raise RuntimeError(
            "The same number of labels, counts, and depths must be specified"
        )

    tp = load_trace(args.input, bin_path=args.trace_processor_shell)

    if tp is None:
        raise ValueError(f"trace {args.input} could not be loaded")

    pdata = {}
    # get data from perfetto
    qr_it = tp.query("SELECT name, depth, category FROM slice")
    # loop over data rows from perfetto
    for row in qr_it:
        if args.categories and row.category not in args.categories:
            continue
        if row.name not in pdata:
            pdata[row.name] = {}
        if row.depth not in pdata[row.name]:
            pdata[row.name][row.depth] = 0
        # accumulate the call-count per name and per depth
        pdata[row.name][row.depth] += 1

    perfetto_data = []
    for name, itr in pdata.items():
        for depth, count in itr.items():
            _e = {}
            _e["label"] = name
            _e["count"] = count
            _e["depth"] = depth
            perfetto_data.append(_e)

    # demo display of data
    if args.print:
        for itr in perfetto_data:
            n = 0 if itr["depth"] < 2 else itr["depth"] - 1
            lbl = "{}{}{}".format(
                "  " * n, "|_" if itr["depth"] > 0 else "", itr["label"]
            )
            print("| {:40} | {:6} | {:6} |".format(lbl, itr["count"], itr["depth"]))

    ret = 0
    try:
        validate_perfetto(
            perfetto_data,
            args.labels,
            args.counts,
            args.depths,
        )

    except RuntimeError as e:
        print(f"{e}")
        ret = 1

    for key_name, key_count in zip(args.key_names, args.key_counts):
        slice_args = tp.query(
            f"select * from slice join args using (arg_set_id) where key='debug.{key_name}'"
        )
        count = 0
        if args.print:
            print(f"{key_name} (expected: {key_count}):")
        for row in slice_args:
            count += 1
            if args.print:
                for key, val in row.__dict__.items():
                    print(f"  - {key:20} :: {val}")
        print(f"Number of entries with {key_name} = {count} (expected: {key_count})")
        if key_count != count:
            ret = 1

    if ret == 0:
        print(f"{args.input} validated")
    else:
        print(f"Failure validating {args.input}")

    sys.exit(ret)
