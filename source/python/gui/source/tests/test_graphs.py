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

from omnitrace_causal_viewer.tests import page
from omnitrace_causal_viewer.__main__ import causal, create_parser, default_settings
from omnitrace_causal_viewer.parser import (
    parse_files,
    find_causal_files,
    parse_uploaded_file,
    process_data,
    compute_speedups,
    compute_sorts,
)

import subprocess
import sys
import os
import time
import multiprocessing
import pandas as pd
import numpy as np
import pytest
import json
from pathlib import Path

from seleniumwire import webdriver

que = os.path.realpath(os.path.dirname(__file__) + "/..")
sys.path.append(que)

# from source.gui import build_causal_layout
path = Path(__file__).parent.absolute()

workload_dir = os.path.realpath(
    os.path.join(
        path,
        *"../workloads/causal-cpu-omni-fast-func-e2e/causal".split("/"),
    )
)

titles = [
    "Selected Causal Profiles",
    "/home/jose/omnitrace/examples/causal/causal.cpp:165",
    "cpu_fast_func(long, int)",
    "cpu_slow_func(long, int)",
]

samples_df_expected_locations = [
    "/home/jose/omnitrace/examples/causal/causal.cpp:103",
    "/home/jose/omnitrace/examples/causal/causal.cpp:110",
    "/home/jose/omnitrace/examples/causal/causal.cpp:112",
    "/usr/include/c++/9/bits/stl_vector.h:125",
    "/usr/include/c++/9/bits/stl_vector.h:128",
    "/usr/include/c++/9/bits/stl_vector.h:285",
    "/usr/include/c++/9/ext/string_conversions.h:83",
    "/usr/include/c++/9/ext/string_conversions.h:84",
    "/usr/include/c++/9/ext/string_conversions.h:85",
]

samples_df_expected_counts = [138, 276, 138, 138, 138, 138, 3312, 414, 690]

input_files = find_causal_files(
    [workload_dir], default_settings["verbose"], default_settings["recursive"]
)

expected_histogram = {
    "/home/jose/omnitrace/examples/causal/causal.cpp:153": 22764,
    "/home/jose/omnitrace/examples/causal/causal.cpp:155": 91056,
    "/home/jose/omnitrace/examples/causal/causal.cpp:157": 68292,
    "/home/jose/omnitrace/examples/causal/causal.cpp:82": 4060,
    "/home/jose/omnitrace/examples/causal/causal.cpp:83": 912,
    "/home/jose/omnitrace/examples/causal/causal.cpp:91": 2030,
    "/home/jose/omnitrace/examples/causal/causal.cpp:93": 912,
    "/usr/include/c++/9/bits/alloc_traits.h:468": 2320,
    "/usr/include/c++/9/bits/allocator.h:137": 2030,
    "/usr/include/c++/9/bits/allocator.h:152": 2030,
    "/usr/include/c++/9/bits/basic_string.h:154": 912,
    "/usr/include/c++/9/bits/basic_string.h:161": 912,
    "/usr/include/c++/9/bits/basic_string.h:204": 912,
    "/usr/include/c++/9/bits/basic_string.h:225": 912,
    "/usr/include/c++/9/bits/basic_string.h:226": 912,
    "/usr/include/c++/9/bits/basic_string.h:2305": 912,
    "/usr/include/c++/9/bits/basic_string.h:233": 912,
    "/usr/include/c++/9/bits/basic_string.h:235": 912,
    "/usr/include/c++/9/bits/basic_string.h:240": 912,
    "/usr/include/c++/9/bits/basic_string.h:6512": 912,
    "/usr/include/c++/9/bits/basic_string.h:661": 912,
    "/usr/include/c++/9/bits/move.h:74": 912,
    "/usr/include/c++/9/bits/random.h:1606": 912,
    "/usr/include/c++/9/bits/unique_ptr.h:147": 912,
    "/usr/include/c++/9/ext/new_allocator.h:114": 2030,
    "/usr/include/c++/9/ext/new_allocator.h:119": 2320,
    "/usr/include/c++/9/ext/new_allocator.h:128": 5220,
    "/usr/include/c++/9/ext/new_allocator.h:80": 2030,
    "/usr/include/c++/9/ext/new_allocator.h:89": 2030,
    "/usr/include/c++/9/ext/string_conversions.h:63": 4930,
    "/usr/include/c++/9/ext/string_conversions.h:64": 3770,
    "/usr/include/c++/9/ext/string_conversions.h:80": 3190,
    "/usr/include/c++/9/ext/string_conversions.h:83": 6960,
    "/usr/include/c++/9/thread:130": 2900,
    "/usr/include/c++/9/thread:82": 2320,
    "/usr/include/c++/9/tuple:132": 912,
    "/usr/include/c++/9/tuple:133": 2900,
    "/usr/include/x86_64-linux-gnu/bits/stdio2.h:105": 2030,
    "/usr/include/x86_64-linux-gnu/bits/stdio2.h:107": 6380,
    "cpu_slow_func(long, int)": 22764,
}

# sparse testing
top_df_expected_program_speedup = [0.0, -1.7623]
top_df_expected_speedup_err = [0.0264, 0.3931]
top_df_expected_impact_sum = np.full(2, -41.6965)
top_df_expected_impact_avg = np.full(2, -13.8988)
top_df_expected_impact_err = np.full(2, 3.6046)
top_df_expected_point_count = np.full(2, 4.0)

mid_df_expected_program_speedup = [0.0, -1.4123]
mid_df_expected_speedup_err = [0.0407, 0.2638]
mid_df_expected_impact_sum = np.full(2, -37.3877)
mid_df_expected_impact_avg = np.full(2, -12.4626)
mid_df_expected_impact_err = np.full(2, 3.8331)
mid_df_expected_point_count = np.full(2, 4.0)

bot_df_expected_program_speedup = [0.0, 10.3991]
bot_df_expected_speedup_err = [0.9115, 0.9072]
bot_df_expected_impact_sum = np.full(2, 385.195)
bot_df_expected_impact_avg = np.full(2, 128.3983)
bot_df_expected_impact_err = np.full(2, 56.9176)
bot_df_expected_point_count = np.full(2, 4.0)


def test_find_causal_files_valid_directory():
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments.coz"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]
    file_names_recursive = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments.coz"),
        os.path.join(workload_dir, *"part2/experiments2.json".split("/")),
        os.path.join(workload_dir, *"part2/experiments1.json".split("/")),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]

    file_names = sorted(file_names)
    file_names_recursive = sorted(file_names_recursive)

    # given a valid directory
    files_found = sorted(
        find_causal_files([workload_dir], default_settings["verbose"], False)
    )

    assert len(files_found) == len(file_names)
    assert files_found == file_names

    # given invalid directory
    with pytest.raises(Exception):
        find_causal_files(["nonsense"], default_settings["verbose"], False)

    # given valid directory with recursive
    files_found = sorted(
        find_causal_files([workload_dir], default_settings["verbose"], True)
    )

    assert len(files_found) == len(file_names_recursive)
    assert files_found == file_names_recursive

    # given invalid directory with recursive
    with pytest.raises(Exception):
        find_causal_files(["nonsense"], default_settings["verbose"], True)


def test_parse_files_default():
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]

    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == samples_df_expected_counts).all()

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()

    # assert expected point count
    assert (top_df["point count"].to_numpy() == top_df_expected_point_count).all()

    middle_df = results_df[
        results_df["idx"]
        == ("causal-cpu-omni", "/home/jose/omnitrace/examples/causal/causal.cpp:165")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        middle_df["program speedup"].to_numpy() == mid_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (middle_df["speedup err"].to_numpy() == mid_df_expected_speedup_err).all()

    # assert exoected impact sum
    assert (middle_df["impact sum"].to_numpy() == mid_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (middle_df["impact avg"].to_numpy() == mid_df_expected_impact_avg).all()

    # assert expected impact err
    assert (middle_df["impact err"].to_numpy() == mid_df_expected_impact_err).all()

    # assert expected point count
    assert (middle_df["point count"].to_numpy() == mid_df_expected_point_count).all()

    bottom_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_slow_func(long, int)")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        bottom_df["program speedup"].to_numpy() == bot_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (bottom_df["speedup err"].to_numpy() == bot_df_expected_speedup_err).all()

    assert (bottom_df["impact sum"].to_numpy() == bot_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (bottom_df["impact avg"].to_numpy() == bot_df_expected_impact_avg).all()

    # assert expected impact err
    assert (bottom_df["impact err"].to_numpy() == bot_df_expected_impact_err).all()

    # assert expected point count
    assert (bottom_df["point count"].to_numpy() == bot_df_expected_point_count).all()


def test_parse_files_valid_directory():
    # test given valid experiment
    file_names = [os.path.join(workload_dir, "experiments.json")]
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        "fast",
        default_settings["progress_points"],
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    _samples_df_expected_counts = [
        152,
        304,
        152,
        152,
        152,
        152,
        3648,
        456,
        760,
    ]

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == _samples_df_expected_counts).all()

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()

    # assert expected point count
    assert (top_df["point count"].to_numpy() == top_df_expected_point_count).all()

    bottom_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][-1:].round(4)

    results_df_expected_program_speedup = [-1.6489]
    results_df_expected_speedup_err = [1.1804]
    results_df_expected_impact_sum = [-41.6965]
    results_df_expected_impact_avg = [-13.8988]
    results_df_expected_impact_err = [3.6046]
    results_df_expected_point_count = [4.0]

    # assert expected speedup err
    assert (
        bottom_df["program speedup"].to_numpy() == results_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (bottom_df["speedup err"].to_numpy() == results_df_expected_speedup_err).all()

    assert (bottom_df["impact sum"].to_numpy() == results_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (bottom_df["impact avg"].to_numpy() == results_df_expected_impact_avg).all()

    # assert expected impact err
    assert (bottom_df["impact err"].to_numpy() == results_df_expected_impact_err).all()

    # assert expected point count
    assert bottom_df["point count"].to_numpy() == results_df_expected_point_count


def test_parse_files_invalid_experiment():
    ############################################################

    # test given invalid experiment
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        "this_is_my_invalid_regex",
        default_settings["progress_points"],
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )
    samples_df_expected_locations = [
        "0x00005555f6213863 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
        "0x00005555f62138e0 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
        "0x00005555f6213f1e :: _start",
        "0x00005600f87738e0 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
        "0x00005600f8773f1e :: _start",
        "0x000056075b7a6863 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
    ]

    file_names = [os.path.join(workload_dir, "experiments.coz")]

    samples_df_expected_counts = [4, 2, 6, 3, 4, 4]

    assert sorted(file_names_run) == sorted(file_names)
    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == samples_df_expected_counts).all()

    results_df = results_df.round(4)
    # returns only .coz outputs since filtering is done in process_data
    expected_points = np.full(4, "cpu_fast_func(long, int)")
    expected_speedup = np.array([0.0, 10.0, 20.0, 30.0])
    expected_progress = np.array([0.0, -1.7623, -1.5829, -1.6489])

    assert (results_df["point"].to_numpy() == expected_points).all()

    assert (results_df["point"].to_numpy() == expected_points).all()
    assert (results_df["speedup"].to_numpy() == expected_speedup).all()
    assert (results_df["progress_speedup"].to_numpy() == expected_progress).all()


def test_parse_files_valid_progress_regex():
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]

    # test given valid progress_point regex
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        "cpu",
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    # sparse testing
    results_df_expected_program_speedup = [0.0, -1.7623]
    results_df_expected_speedup_err = [0.0264, 0.3931]
    results_df_expected_impact_sum = np.full(2, -41.6965)
    results_df_expected_impact_avg = np.full(2, -13.8988)
    results_df_expected_impact_err = np.full(2, 3.6046)
    results_df_expected_point_count = np.full(2, 4.0)

    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    # samples_df_expected_counts = [152, 304, 152, 152, 152, 152, 3648, 456, 760]

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == samples_df_expected_counts).all()

    # assert expected speedup err
    assert (
        top_df["program speedup"].to_numpy() == results_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == results_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == results_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == results_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == results_df_expected_impact_err).all()

    # assert expected point count
    assert (top_df["point count"].to_numpy() == results_df_expected_point_count).all()

    middle_df = results_df[
        results_df["idx"]
        == ("causal-cpu-omni", "/home/jose/omnitrace/examples/causal/causal.cpp:165")
    ][:2].round(4)

    results_df_expected_program_speedup = [0.0, -1.4123]
    results_df_expected_speedup_err = [0.0407, 0.2638]
    results_df_expected_impact_sum = np.full(2, -37.3877)
    results_df_expected_impact_avg = np.full(2, -12.4626)
    results_df_expected_impact_err = np.full(2, 3.8331)
    results_df_expected_point_count = np.full(2, 4.0)

    # assert expected speedup err
    assert (
        middle_df["program speedup"].to_numpy() == mid_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (middle_df["speedup err"].to_numpy() == mid_df_expected_speedup_err).all()

    # assert exoected impact sum
    assert (middle_df["impact sum"].to_numpy() == mid_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (middle_df["impact avg"].to_numpy() == mid_df_expected_impact_avg).all()

    # assert expected impact err
    assert (middle_df["impact err"].to_numpy() == mid_df_expected_impact_err).all()

    # assert expected point count
    assert (middle_df["point count"].to_numpy() == mid_df_expected_point_count).all()

    bottom_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_slow_func(long, int)")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        bottom_df["program speedup"].to_numpy() == bot_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (bottom_df["speedup err"].to_numpy() == bot_df_expected_speedup_err).all()

    assert (bottom_df["impact sum"].to_numpy() == bot_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (bottom_df["impact avg"].to_numpy() == bot_df_expected_impact_avg).all()

    # assert expected impact err
    assert (bottom_df["impact err"].to_numpy() == bot_df_expected_impact_err).all()

    # assert expected point count
    assert (bottom_df["point count"].to_numpy() == bot_df_expected_point_count).all()


def test_parse_files_invalid_progress_regex():
    # test given invalid progress_point regex
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        "this_is_my_invalid_regex",
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    samples_df_expected_locations = [
        "0x00005555f6213863 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
        "0x00005555f62138e0 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
        "0x00005555f6213f1e :: _start",
        "0x00005600f87738e0 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
        "0x00005600f8773f1e :: _start",
        "0x000056075b7a6863 :: /home/jose/omnitrace/examples/causal/causal.cpp:71",
    ]

    file_names = [os.path.join(workload_dir, "experiments.coz")]

    results_df = results_df.round(4)
    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    expected_points = np.full(4, "cpu_fast_func(long, int)")
    expected_speedup = np.array([0.0, 10.0, 20.0, 30.0])
    expected_progress = np.array([0.0, -1.7623, -1.5829, -1.6489])
    expected_samples_count = np.array([4, 2, 6, 3, 4, 4])

    assert (results_df["point"].to_numpy() == expected_points).all()

    assert (results_df["point"].to_numpy() == expected_points).all()
    assert (results_df["speedup"].to_numpy() == expected_speedup).all()
    assert (results_df["progress_speedup"].to_numpy() == expected_progress).all()
    assert sorted(file_names_run) == sorted(file_names)
    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == expected_samples_count).all()


def test_parse_files_valid_speedup():
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]

    # test given valid speedup
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [0, 10],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    # sparse testing
    results_df_expected_impact_sum = np.full(2, -8.8117)
    results_df_expected_impact_avg = np.full(2, -8.8117)
    results_df_expected_impact_err = np.full(2, 0)
    results_df_expected_point_count = np.full(2, 2.0)

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_list()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_list()

    assert samples_df_locations == samples_df_expected_locations
    assert samples_df_counts == samples_df_expected_counts

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == results_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == results_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == results_df_expected_impact_err).all()

    # assert expected point count
    assert (top_df["point count"].to_numpy() == results_df_expected_point_count).all()

    middle_df = results_df[
        results_df["idx"]
        == ("causal-cpu-omni", "/home/jose/omnitrace/examples/causal/causal.cpp:165")
    ][:2].round(4)

    results_df_expected_impact_sum = np.full(2, -7.0613)
    results_df_expected_impact_avg = np.full(2, -7.0613)
    results_df_expected_impact_err = np.full(2, 0)
    results_df_expected_point_count = np.full(2, 2.0)

    # assert expected speedup err
    assert (
        middle_df["program speedup"].to_numpy() == mid_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (middle_df["speedup err"].to_numpy() == mid_df_expected_speedup_err).all()

    # assert exoected impact sum
    assert (middle_df["impact sum"].to_numpy() == results_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (middle_df["impact avg"].to_numpy() == results_df_expected_impact_avg).all()

    # assert expected impact err
    assert (middle_df["impact err"].to_numpy() == results_df_expected_impact_err).all()

    # assert expected point count
    assert (middle_df["point count"].to_numpy() == results_df_expected_point_count).all()

    bottom_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_slow_func(long, int)")
    ][:2].round(4)

    results_df_expected_impact_sum = np.full(2, 51.9953)
    results_df_expected_impact_avg = np.full(2, 51.9953)
    results_df_expected_impact_err = np.full(2, 0)
    results_df_expected_point_count = np.full(2, 2.0)

    # assert expected speedup err
    assert (
        bottom_df["program speedup"].to_numpy() == bot_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (bottom_df["speedup err"].to_numpy() == bot_df_expected_speedup_err).all()

    assert (bottom_df["impact sum"].to_numpy() == results_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (bottom_df["impact avg"].to_numpy() == results_df_expected_impact_avg).all()

    # assert expected impact err
    assert (bottom_df["impact err"].to_numpy() == results_df_expected_impact_err).all()

    # assert expected point count
    assert (bottom_df["point count"].to_numpy() == results_df_expected_point_count).all()


def test_parse_files_invalid_speedup():
    # test given invalid speedup
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]

    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [12, 14],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == samples_df_expected_counts).all()

    assert results_df.empty


def test_parse_files_valid_min_points():
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]

    ##############################################################################################
    # test given valid min points
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        1,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == samples_df_expected_counts).all()

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()

    # assert expected point count
    assert (top_df["point count"].to_numpy() == top_df_expected_point_count).all()

    middle_df = results_df[
        results_df["idx"]
        == ("causal-cpu-omni", "/home/jose/omnitrace/examples/causal/causal.cpp:165")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        middle_df["program speedup"].to_numpy() == mid_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (middle_df["speedup err"].to_numpy() == mid_df_expected_speedup_err).all()

    # assert exoected impact sum
    assert (middle_df["impact sum"].to_numpy() == mid_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (middle_df["impact avg"].to_numpy() == mid_df_expected_impact_avg).all()

    # assert expected impact err
    assert (middle_df["impact err"].to_numpy() == mid_df_expected_impact_err).all()

    # assert expected point count
    assert (middle_df["point count"].to_numpy() == mid_df_expected_point_count).all()

    bottom_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_slow_func(long, int)")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        bottom_df["program speedup"].to_numpy() == bot_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (bottom_df["speedup err"].to_numpy() == bot_df_expected_speedup_err).all()

    assert (bottom_df["impact sum"].to_numpy() == bot_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (bottom_df["impact avg"].to_numpy() == bot_df_expected_impact_avg).all()

    # assert expected impact err
    assert (bottom_df["impact err"].to_numpy() == bot_df_expected_impact_err).all()

    # assert expected point count
    assert (bottom_df["point count"].to_numpy() == bot_df_expected_point_count).all()


def test_parse_files_high_min_points():
    file_names = [
        os.path.join(workload_dir, "experiments.json"),
        os.path.join(workload_dir, "experiments3.json"),
        os.path.join(workload_dir, "experiments4.json"),
    ]
    ###################################################################################
    # test given too high min points
    results_df, samples_df, file_names_run = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        1000,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )
    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    assert sorted(file_names_run) == sorted(file_names)

    samples_df_locations = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["location"].to_numpy()
    samples_df_counts = pd.concat(
        [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
    )["count"].to_numpy()

    assert (samples_df_locations == samples_df_expected_locations).all()
    assert (samples_df_counts == samples_df_expected_counts).all()

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()

    # assert expected point count
    assert (top_df["point count"].to_numpy() == top_df_expected_point_count).all()

    middle_df = results_df[
        results_df["idx"]
        == ("causal-cpu-omni", "/home/jose/omnitrace/examples/causal/causal.cpp:165")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        middle_df["program speedup"].to_numpy() == mid_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (middle_df["speedup err"].to_numpy() == mid_df_expected_speedup_err).all()

    # assert exoected impact sum
    assert (middle_df["impact sum"].to_numpy() == mid_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (middle_df["impact avg"].to_numpy() == mid_df_expected_impact_avg).all()

    # assert expected impact err
    assert (middle_df["impact err"].to_numpy() == mid_df_expected_impact_err).all()

    # assert expected point count
    assert (middle_df["point count"].to_numpy() == mid_df_expected_point_count).all()

    bottom_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_slow_func(long, int)")
    ][:2].round(4)

    # assert expected speedup err
    assert (
        bottom_df["program speedup"].to_numpy() == bot_df_expected_program_speedup
    ).all()

    # assert expected speedup err
    assert (bottom_df["speedup err"].to_numpy() == bot_df_expected_speedup_err).all()

    assert (bottom_df["impact sum"].to_numpy() == bot_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (bottom_df["impact avg"].to_numpy() == bot_df_expected_impact_avg).all()

    # assert expected impact err
    assert (bottom_df["impact err"].to_numpy() == bot_df_expected_impact_err).all()

    # assert expected point count
    assert (bottom_df["point count"].to_numpy() == bot_df_expected_point_count).all()


def test_process_data():
    # test with valid data
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())
        dict_data = {}
        data = process_data(dict_data, _data, ".*", ".*")
        assert list(dict_data.keys()) == ["cpu_fast_func(long, int)"]
        assert list(data.keys()) == ["cpu_fast_func(long, int)"]

        data = process_data({}, _data, ".*", "fast")
        assert list(dict_data.keys()) == ["cpu_fast_func(long, int)"]
        assert list(data.keys()) == []

        data = process_data({}, _data, "fast", ".*")
        assert list(dict_data.keys()) == ["cpu_fast_func(long, int)"]
        assert list(data.keys()) == ["cpu_fast_func(long, int)"]

        data = process_data({}, _data, ".*", "impl")
        assert list(dict_data.keys()) == ["cpu_fast_func(long, int)"]
        assert list(data.keys()) == []

        data = process_data({}, _data, "impl", ".*")
        assert list(dict_data.keys()) == ["cpu_fast_func(long, int)"]
        assert list(data.keys()) == []


def test_compute_speedups_verb_3():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )

        # Testing verbosity
        results_df = compute_speedups(
            dict_data, [], default_settings["min_points"], [], 3
        )

        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_verb_2():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # Testing verbosity
        results_df = compute_speedups(
            dict_data, [], default_settings["min_points"], [], 2
        )
        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_verb_1():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # Testing verbosity
        results_df = compute_speedups(
            dict_data, [], default_settings["min_points"], [], 1
        )

        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_verb_0():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )

        # Testing verbosity
        results_df = compute_speedups(
            dict_data, [], default_settings["min_points"], [], 0
        )

        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_verb_4():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # Testing verbosity
        results_df = compute_speedups(
            dict_data, [], default_settings["min_points"], [], 4
        )

        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_high_min_points():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # min points too high
        results_df = compute_speedups(dict_data, [], 247, [], 3)

        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_min_points_0():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # min points 0
        results_df = compute_speedups(dict_data, [], 0, [], 3)

        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_min_points_1():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # min points 1
        results_df = compute_speedups(dict_data, [], 1, [], 3)
        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_empty_dict():
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # empty dict_data
        results_df = compute_speedups({}, [], 0, [], 3)
        assert results_df.empty


def test_compute_speedups_validate_file():
    experiment_regex = "fast"
    progress_point_regex = "cpu"
    virtual_speedup = "10"
    expected_speedup = "0.8"
    tolerance = "50"
    validate = [
        experiment_regex,
        progress_point_regex,
        virtual_speedup,
        expected_speedup,
        tolerance,
    ]

    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())

        dict_data = {}
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # min points too high
        results_df = compute_speedups(dict_data, [], 0, validate, 3)
        top_df = results_df[
            results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
        ][:2].round(4)

        # assert expected speedup err
        assert (
            top_df["program speedup"].to_numpy() == top_df_expected_program_speedup
        ).all()

        # assert expected speedup err
        assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

        assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

        # assert expected impact avg
        assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

        # assert expected impact err
        assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_speedups_validate_multi_file():
    experiment_regex = "fast"
    progress_point_regex = "cpu"
    virtual_speedup = "10"
    expected_speedup = "0.8"
    tolerance = "50"
    validate = [
        experiment_regex,
        progress_point_regex,
        virtual_speedup,
        expected_speedup,
        tolerance,
    ]

    dict_data = {}
    with open(os.path.join(workload_dir, "experiments.json")) as file:
        _data = json.loads(file.read())
        dict_data[os.path.join(workload_dir, "experiments.json")] = process_data(
            {}, _data, ".*", ".*"
        )
    with open(os.path.join(workload_dir, "experiments3.json")) as file:
        _data = json.loads(file.read())
        dict_data[os.path.join(workload_dir, "experiments3.json")] = process_data(
            {}, _data, ".*", ".*"
        )
    with open(os.path.join(workload_dir, "experiments4.json")) as file:
        _data = json.loads(file.read())
        dict_data[os.path.join(workload_dir, "experiments4.json")] = process_data(
            {}, _data, ".*", ".*"
        )
        # min points too high
    results_df = compute_speedups(dict_data, [], 0, validate, 3)
    top_df = results_df[
        results_df["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")
    ][:2].round(4)

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()


def test_compute_sorts():
    file_name = os.path.join(workload_dir, "experiments.json")
    with open(file_name) as file:
        _data = json.loads(file.read())
        dict_data = {}
        dict_data = {file_name: process_data(dict_data, _data, ".*", ".*")}

        data = compute_sorts(compute_speedups(dict_data))

        expected_speedup = np.full(4, 0.0)
        expected_point_count = np.full(4, 4.0)

        assert (data["max speedup"].to_numpy() == expected_speedup).all()
        assert (data["min speedup"].to_numpy() == expected_speedup).all()
        assert (data["point count"].to_numpy() == expected_point_count).all()

    results_df = pd.DataFrame()
    files = find_causal_files([workload_dir], default_settings["verbose"], True)
    json_files = [x for x in filter(lambda y: y.endswith(".json"), files)]
    for file in json_files:
        with open(file, "r") as j:
            _data = json.load(j)
            dict_data = {}
            # make sure the JSON is an omnitrace causal JSON
            if "omnitrace" not in _data or "causal" not in _data["omnitrace"]:
                continue
            dict_data[file] = process_data({}, _data, ".*", ".*")

            results_df = pd.concat(
                [
                    results_df,
                    compute_sorts(
                        compute_speedups(
                            dict_data,
                            [],
                            0,
                            [],
                            False,
                        )
                    ),
                ]
            )

    expected_speedup = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    expected_point_count = [
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        5.0,
        5.0,
        5.0,
        5.0,
        5.0,
        2.0,
        2.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        2.0,
        2.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
    ]

    assert results_df["max speedup"].to_list() == expected_speedup
    assert results_df["min speedup"].to_list() == expected_speedup
    assert results_df["point count"].to_list() == expected_point_count


def test_parse_uploaded_file():
    file_name = os.path.join(workload_dir, "experiments.json")
    with open(file_name) as file:
        _data = file.read()
        data, samples_df = parse_uploaded_file(file_name, _data)

        top_df = data[data["idx"] == ("causal-cpu-omni", "cpu_fast_func(long, int)")][
            :2
        ].round(4)

        samples_df_locations = pd.concat(
            [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
        )["location"].to_list()
        samples_df_counts = pd.concat(
            [samples_df[0:3], samples_df[100:103], samples_df[150:153]]
        )["count"].to_list()

    # assert expected speedup err
    assert (top_df["program speedup"].to_numpy() == top_df_expected_program_speedup).all()

    # assert expected speedup err
    assert (top_df["speedup err"].to_numpy() == top_df_expected_speedup_err).all()

    assert (top_df["impact sum"].to_numpy() == top_df_expected_impact_sum).all()

    # assert expected impact avg
    assert (top_df["impact avg"].to_numpy() == top_df_expected_impact_avg).all()

    # assert expected impact err
    assert (top_df["impact err"].to_numpy() == top_df_expected_impact_err).all()

    _samples_df_expected_counts = [
        152,
        304,
        152,
        152,
        152,
        152,
        3648,
        456,
        760,
    ]

    assert samples_df_locations == samples_df_expected_locations
    assert samples_df_counts == _samples_df_expected_counts


def set_up(ip_addr="localhost", ip_port="8051"):
    # works for linux, no browser pops up
    fireFoxOptions = webdriver.FirefoxOptions()
    fireFoxOptions.add_argument("--headless")
    browser = webdriver.Firefox(options=fireFoxOptions)
    browser.get("http://" + ip_addr + ":" + ip_port + "/")
    return browser


# test order of chart titles
def test_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(["-w", workload_dir, "-n", "0"])

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    try:
        t.start()
        time.sleep(5)

        driver = set_up()
        main_page = page.MainPage(driver)

        expected_title_set = [
            "Selected Causal Profiles",
            "cpu_slow_func(long, int)",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
        ]
        captured_output = main_page.get_titles()

        driver.quit()
    finally:
        t.terminate()
        t.join()

    assert captured_output == expected_title_set


def test_alphabetical_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(["-w", workload_dir, "-n", "0"])

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    try:
        t.start()
        time.sleep(5)
        driver = set_up()
        main_page = page.MainPage(driver)

        expected_title_set = [
            "Selected Causal Profiles",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
            "cpu_slow_func(long, int)",
        ]

        title_set = main_page.get_alphabetical_titles()
        captured_plot_data = main_page.get_plot_data()
        captured_histogram_data = main_page.get_histogram_data()

        driver.quit()
    finally:
        t.terminate()
        t.join()

    assert (
        np.array(captured_plot_data[0]["error_y"]["array"]).round(4)
        == [0.9115, 0.9072, 0.9204, 0.3939]
    ).all()
    assert captured_plot_data[0]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[0]["y"]).round(4) == [0.0, 10.3991, 18.533, 19.1749]
    ).all()
    assert (
        np.array(captured_plot_data[2]["error_y"]["array"]).round(4)
        == [0.0264, 0.3931, 1.271, 1.1804]
    ).all()
    assert captured_plot_data[2]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[2]["y"]).round(4) == [0.0, -1.7623, -1.5829, -1.6489]
    ).all()

    assert title_set == expected_title_set
    assert captured_histogram_data == expected_histogram


def test_max_speedup_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(["-w", workload_dir, "-n", "0"])

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    try:
        t.start()
        time.sleep(5)
        driver = set_up()

        main_page = page.MainPage(driver)
        captured_output = main_page.get_max_speedup_titles()
        captured_histogram_data = main_page.get_histogram_data()
        captured_plot_data = main_page.get_plot_data()
        expected_title_set = [
            "Selected Causal Profiles",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
            "cpu_slow_func(long, int)",
        ]

        driver.quit()
    finally:
        t.terminate()
        t.join()

    assert (
        np.array(captured_plot_data[0]["error_y"]["array"]).round(4)
        == [0.9115, 0.9072, 0.9204, 0.3939]
    ).all()
    assert captured_plot_data[0]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[0]["y"]).round(4) == [0.0, 10.3991, 18.533, 19.1749]
    ).all()
    assert (
        np.array(captured_plot_data[2]["error_y"]["array"]).round(4)
        == [0.0264, 0.3931, 1.271, 1.1804]
    ).all()
    assert captured_plot_data[2]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[2]["y"]).round(4) == [0.0, -1.7623, -1.5829, -1.6489]
    ).all()

    assert captured_output == expected_title_set
    assert captured_histogram_data == expected_histogram


def test_min_speedup_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(["-w", workload_dir, "-n", "0"])

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    try:
        t.start()
        time.sleep(5)
        driver = set_up()

        main_page = page.MainPage(driver)

        expected_title_set = [
            "Selected Causal Profiles",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
            "cpu_slow_func(long, int)",
        ]
        captured_output = main_page.get_min_speedup_titles()
        captured_histogram_data = main_page.get_histogram_data()
        captured_plot_data = main_page.get_plot_data()

        driver.quit()
    finally:
        t.terminate()
        t.join()

    assert (
        np.array(captured_plot_data[0]["error_y"]["array"]).round(4)
        == [0.9115, 0.9072, 0.9204, 0.3939]
    ).all()
    assert captured_plot_data[0]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[0]["y"]).round(4) == [0.0, 10.3991, 18.533, 19.1749]
    ).all()
    assert (
        np.array(captured_plot_data[2]["error_y"]["array"]).round(4)
        == [0.0264, 0.3931, 1.271, 1.1804]
    ).all()
    assert captured_plot_data[2]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[2]["y"]).round(4) == [0.0, -1.7623, -1.5829, -1.6489]
    ).all()

    assert captured_output == expected_title_set
    assert captured_histogram_data == expected_histogram


def test_impact_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(["-w", workload_dir, "-n", "0"])

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    try:
        t.start()

        time.sleep(5)
        driver = set_up()

        main_page = page.MainPage(driver)

        expected_title_set = [
            "Selected Causal Profiles",
            "cpu_slow_func(long, int)",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
        ]
        captured_output = main_page.get_impact_titles()
        captured_histogram_data = main_page.get_histogram_data()
        captured_plot_data = main_page.get_plot_data()

        driver.quit()
    finally:
        t.terminate()
        t.join()

    assert (
        np.array(captured_plot_data[0]["error_y"]["array"]).round(4)
        == [0.9115, 0.9072, 0.9204, 0.3939]
    ).all()
    assert captured_plot_data[0]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[0]["y"]).round(4) == [0.0, 10.3991, 18.533, 19.1749]
    ).all()
    assert (
        np.array(captured_plot_data[2]["error_y"]["array"]).round(4)
        == [0.0264, 0.3931, 1.271, 1.1804]
    ).all()
    assert captured_plot_data[2]["x"] == [0, 10, 20, 30]
    assert (
        np.array(captured_plot_data[2]["y"]).round(4) == [0.0, -1.7623, -1.5829, -1.6489]
    ).all()
    assert captured_histogram_data == expected_histogram

    assert captured_output == expected_title_set


def test_min_points_slider():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    try:
        t.start()
        time.sleep(5)

        driver = set_up()
        main_page = page.MainPage(driver)
        expected_title_set = [
            {"num points": 9, "titles": ["Selected Causal Profiles"]},
            {"num points": 8, "titles": ["Selected Causal Profiles"]},
            {"num points": 7, "titles": ["Selected Causal Profiles"]},
            {"num points": 6, "titles": ["Selected Causal Profiles"]},
            {"num points": 5, "titles": ["Selected Causal Profiles"]},
            {
                "num points": 4,
                "titles": [
                    "Selected Causal Profiles",
                    "cpu_slow_func(long, int)",
                    "/home/jose/omnitrace/examples/causal/causal.cpp:165",
                    "cpu_fast_func(long, int)",
                ],
            },
            {
                "num points": 3,
                "titles": [
                    "Selected Causal Profiles",
                    "cpu_slow_func(long, int)",
                    "/home/jose/omnitrace/examples/causal/causal.cpp:165",
                    "cpu_fast_func(long, int)",
                ],
            },
            {
                "num points": 2,
                "titles": [
                    "Selected Causal Profiles",
                    "cpu_slow_func(long, int)",
                    "/home/jose/omnitrace/examples/causal/causal.cpp:165",
                    "cpu_fast_func(long, int)",
                ],
            },
            {
                "num points": 1,
                "titles": [
                    "Selected Causal Profiles",
                    "cpu_slow_func(long, int)",
                    "/home/jose/omnitrace/examples/causal/causal.cpp:165",
                    "cpu_fast_func(long, int)",
                ],
            },
            {
                "num points": 0,
                "titles": [
                    "Selected Causal Profiles",
                    "cpu_slow_func(long, int)",
                    "/home/jose/omnitrace/examples/causal/causal.cpp:165",
                    "cpu_fast_func(long, int)",
                ],
            },
        ]
        captured_output = main_page.get_min_points_titles()
        captured_histogram_data = main_page.get_histogram_data()

        # captured_plot_data = main_page.get_plot_data()

        driver.quit()

    finally:
        t.terminate()
        t.join()

    assert captured_output == expected_title_set
    assert captured_histogram_data == expected_histogram


def test_verbose_gui_flag_1():
    t = subprocess.Popen(
        [sys.executable, "-m", "source", "-w", workload_dir, "--verbose", "1", "-n", "0"],
        stdout=subprocess.PIPE,
    )

    try:
        time.sleep(5)
        driver = set_up()
        main_page = page.MainPage(driver)

        expected_title_set = [
            "Selected Causal Profiles",
            "cpu_slow_func(long, int)",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
        ]
        captured_title_set = main_page.get_titles()
        captured_output = t.communicate(timeout=15)[0].decode("utf-8")
        driver.quit()

    finally:
        t.terminate()

    assert captured_title_set == expected_title_set
    assert captured_output


def test_verbose_gui_flag_2():
    t = subprocess.Popen(
        [sys.executable, "-m", "source", "-w", workload_dir, "--verbose", "2", "-n", "0"],
        stdout=subprocess.PIPE,
    )

    try:
        expected_title_set = [
            "Selected Causal Profiles",
            "cpu_slow_func(long, int)",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
        ]
        time.sleep(5)
        driver = set_up()
        main_page = page.MainPage(driver)
        captured_title_set = main_page.get_titles()
        captured_output = t.communicate(timeout=15)[0].decode("utf-8")
        driver.quit()
    finally:
        t.terminate()

    assert captured_output
    assert captured_title_set == expected_title_set


def test_verbose_gui_flag_3():
    expected_title_set = [
        "Selected Causal Profiles",
        "cpu_slow_func(long, int)",
        "/home/jose/omnitrace/examples/causal/causal.cpp:165",
        "cpu_fast_func(long, int)",
    ]

    t = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "source",
            "-w",
            workload_dir,
            "--verbose",
            "3",
            "-n",
            "0",
        ],
        stdout=subprocess.PIPE,
    )

    try:
        time.sleep(5)
        driver = set_up()
        main_page = page.MainPage(driver)

        captured_title_set = main_page.get_titles()
        driver.quit()
    finally:
        t.terminate()

    captured_output = t.communicate(timeout=15)[0].decode("utf-8")

    assert captured_title_set == expected_title_set
    assert captured_output


def test_ip_port_flag():
    t = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "source",
            "-w",
            workload_dir,
            "--port",
            "8052",
        ],
        stdout=subprocess.PIPE,
    )

    try:
        time.sleep(5)
        driver = set_up(ip_port="8052")
        main_page = page.MainPage(driver)

        expected_title_set = [
            "Selected Causal Profiles",
            "cpu_slow_func(long, int)",
            "/home/jose/omnitrace/examples/causal/causal.cpp:165",
            "cpu_fast_func(long, int)",
        ]
        expected_output = "running on http://0.0.0.0:8052"

        captured_title_set = main_page.get_titles()
        driver.quit()

    finally:
        t.terminate()

    captured_output = t.communicate(timeout=15)[0].decode("utf-8")

    assert captured_title_set == expected_title_set
    assert expected_output in captured_output

    return True
