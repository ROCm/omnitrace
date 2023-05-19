# import unittest
from seleniumwire import webdriver
import page
from pyvirtualdisplay import Display

# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# from selenium.webdriver.chrome.options import Options
import subprocess
import sys, os
import time
import multiprocessing

que = os.path.realpath(os.path.dirname(__file__) + "/..")
sys.path.append(que)
import pytest
from source.gui import build_causal_layout
from source.__main__ import causal, create_parser, default_settings
from source.parser import parse_files, find_causal_files, set_num_stddev, parse_uploaded_file, process_data, compute_speedups
from threading import Thread


import json 

# import logging
# LOGGER = logging.getLogger(__name__)

from pathlib import Path

path = Path(__file__).parent.absolute()


workload_dir  = os.path.realpath(os.path.join(path, *"../workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal".split('/')))
print(workload_dir)

workload_file = os.path.join(workload_dir, "experiments.json")
print(workload_file)




def test_parse_files():
    input_files = find_causal_files([workload_dir], default_settings["verbose"], default_settings["recursive"])
    results_df, samples_df, file_names = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files"
    if test_name in expected_results:
        expected_results_df = expected_results[test_name]["results_df"]
        expected_samples_df = expected_results[test_name]["samples_df"]
        expected_file_names = expected_results[test_name]["file_names"]

    else:
        expected_results[test_name]={}
        expected_results_df = expected_results[test_name]["results_df"] = results_df.to_json(orient="split")
        expected_samples_df = expected_results[test_name]["samples_df"] = samples_df.to_json(orient="split")
        expected_file_names = expected_results[test_name]["file_names"] = file_names

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df.to_json(orient="split") == expected_results_df)
    assert(samples_df.to_json(orient="split") == expected_samples_df)
    assert(file_names == expected_file_names)

    # test given valid experiment
    results_df_2, samples_df_2, file_names_2 = parse_files(
        input_files,
        "fast",
        default_settings["progress_points"],
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_2"
    if test_name in expected_results:
        expected_results_df_2 = expected_results[test_name]["results_df_2"]
        expected_samples_df_2 = expected_results[test_name]["samples_df_2"]
        expected_file_names_2 = expected_results[test_name]["file_names_2"]

    else:
        expected_results[test_name]={}
        expected_results_df_2 = expected_results[test_name]["results_df_2"] = results_df_2.to_json(orient="split")
        expected_samples_df_2 = expected_results[test_name]["samples_df_2"] = samples_df_2.to_json(orient="split")
        expected_file_names_2 = expected_results[test_name]["file_names_2"] = file_names_2

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_2.to_json(orient="split") == expected_results_df_2)
    assert(samples_df_2.to_json(orient="split") == expected_samples_df_2)
    assert(file_names_2 == expected_file_names_2)

    # test given invalid experiment
    results_df_3, samples_df_3, file_names_3 = parse_files(
        input_files,
        "this_is_my_invalid_regex",
        default_settings["progress_points"],
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_3"
    if test_name in expected_results:
        expected_results_df_3 = expected_results[test_name]["results_df_3"]
        expected_samples_df_3 = expected_results[test_name]["samples_df_3"]
        expected_file_names_3 = expected_results[test_name]["file_names_3"]

    else:
        expected_results[test_name]={}
        expected_results_df_3 = expected_results[test_name]["results_df_3"] = results_df_3.to_json(orient="split")
        expected_samples_df_3= expected_results[test_name]["samples_df_3"] = samples_df_3.to_json(orient="split")
        expected_file_names_3 = expected_results[test_name]["file_names_3"] = file_names_3

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_3.to_json(orient="split") == expected_results_df_3)
    assert(samples_df_3.to_json(orient="split") == expected_samples_df_3)
    assert(file_names_3 == expected_file_names_3)

    # test given valid progress_point regex
    results_df_4, samples_df_4, file_names_4 = parse_files(
        input_files,
        default_settings["experiments"],
        "cpu",
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_4"
    if test_name in expected_results:
        expected_results_df_4 = expected_results[test_name]["results_df_4"]
        expected_samples_df_4 = expected_results[test_name]["samples_df_4"]
        expected_file_names_4 = expected_results[test_name]["file_names_4"]

    else:
        expected_results[test_name]={}
        expected_results_df_4 = expected_results[test_name]["results_df_4"] = results_df_4.to_json(orient="split")
        expected_samples_df_4 = expected_results[test_name]["samples_df_4"] = samples_df_4.to_json(orient="split")
        expected_file_names_4 = expected_results[test_name]["file_names_4"] = file_names_4

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_4.to_json(orient="split") == expected_results_df_4)
    assert(samples_df_4.to_json(orient="split") == expected_samples_df_4)
    assert(file_names_4 == expected_file_names_4)

    # test given invalid progress_point regex
    results_df_5, samples_df_5, file_names_5 = parse_files(
        input_files,
        default_settings["experiments"],
        "this_is_my_invalid_regex",
        [],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_5"
    if test_name in expected_results:
        expected_results_df_5 = expected_results[test_name]["results_df_5"]
        expected_samples_df_5 = expected_results[test_name]["samples_df_5"]
        expected_file_names_5 = expected_results[test_name]["file_names_5"]

    else:
        expected_results[test_name]={}
        expected_results_df_5 = expected_results[test_name]["results_df_5"] = results_df_5.to_json(orient="split")
        expected_samples_df_5 = expected_results[test_name]["samples_df_5"] = samples_df_5.to_json(orient="split")
        expected_file_names_5 = expected_results[test_name]["file_names_5"] = file_names_5

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_5.to_json(orient="split") == expected_results_df_5)
    assert(samples_df_5.to_json(orient="split") == expected_samples_df_5)
    assert(file_names_5 == expected_file_names_5)

    # test given valid speedup
    results_df_6, samples_df_6, file_names_6 = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [0, 10],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_6"
    if test_name in expected_results:
        expected_results_df_6 = expected_results[test_name]["results_df_6"]
        expected_samples_df_6 = expected_results[test_name]["samples_df_6"]
        expected_file_names_6 = expected_results[test_name]["file_names_6"]

    else:
        expected_results[test_name]={}
        expected_results_df_6 = expected_results[test_name]["results_df_6"] = results_df_6.to_json(orient="split")
        expected_samples_df_6 = expected_results[test_name]["samples_df_6"] = samples_df_6.to_json(orient="split")
        expected_file_names_6 = expected_results[test_name]["file_names_6"] = file_names_6

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_6.to_json(orient="split") == expected_results_df_6)
    assert(samples_df_6.to_json(orient="split") == expected_samples_df_6)
    assert(file_names_6 == expected_file_names_6)

    # test given invalid speedup
    results_df_7, samples_df_7, file_names_7 = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [12, 14],
        0,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_7"
    if test_name in expected_results:
        expected_results_df_7 = expected_results[test_name]["results_df_7"]
        expected_samples_df_7 = expected_results[test_name]["samples_df_7"]
        expected_file_names_7 = expected_results[test_name]["file_names_7"]

    else:
        expected_results[test_name]={}
        expected_results_df_7 = expected_results[test_name]["results_df_7"] = results_df_7.to_json(orient="split")
        expected_samples_df_7 = expected_results[test_name]["samples_df_7"] = samples_df_7.to_json(orient="split")
        expected_file_names_7 = expected_results[test_name]["file_names_7"] = file_names_7

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_7.to_json(orient="split") == expected_results_df_7)
    assert(samples_df_7.to_json(orient="split") == expected_samples_df_7)
    assert(file_names_7 == expected_file_names_7)

    # test given valid min points
    results_df_8, samples_df_8, file_names_8 = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        1,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_8"
    if test_name in expected_results:
        expected_results_df_8 = expected_results[test_name]["results_df_8"]
        expected_samples_df_8 = expected_results[test_name]["samples_df_8"]
        expected_file_names_8 = expected_results[test_name]["file_names_8"]

    else:
        expected_results[test_name]={}
        expected_results_df_8 = expected_results[test_name]["results_df_8"] = results_df_8.to_json(orient="split")
        expected_samples_df_8 = expected_results[test_name]["samples_df_8"] = samples_df_8.to_json(orient="split")
        expected_file_names_8 = expected_results[test_name]["file_names_8"] = file_names_8

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_8.to_json(orient="split") == expected_results_df_8)
    assert(samples_df_8.to_json(orient="split") == expected_samples_df_8)
    assert(file_names_8 == expected_file_names_8)

    # test given invalid min points
    results_df_9, samples_df_9, file_names_9 = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        1000,
        [],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_9"
    if test_name in expected_results:
        expected_results_df_9 = expected_results[test_name]["results_df_9"]
        expected_samples_df_9 = expected_results[test_name]["samples_df_9"]
        expected_file_names_9 = expected_results[test_name]["file_names_9"]

    else:
        expected_results[test_name]={}
        expected_results_df_9 = expected_results[test_name]["results_df_9"] = results_df_9.to_json(orient="split")
        expected_samples_df_9 = expected_results[test_name]["samples_df_9"] = samples_df_9.to_json(orient="split")
        expected_file_names_9 = expected_results[test_name]["file_names_9"] = file_names_9

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_9.to_json(orient="split") == expected_results_df_9)
    assert(samples_df_9.to_json(orient="split") == expected_samples_df_9)
    assert(file_names_9 == expected_file_names_9)

    # test given valid validation
    results_df_10, samples_df_10, file_names_10 = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        0,
        ["fast", ".*", "10", "-2","1"],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_10"
    if test_name in expected_results:
        expected_results_df_10 = expected_results[test_name]["results_df_10"]
        expected_samples_df_10 = expected_results[test_name]["samples_df_10"]
        expected_file_names_10 = expected_results[test_name]["file_names_10"]

    else:
        expected_results[test_name]={}
        expected_results_df_10 = expected_results[test_name]["results_df_10"] = results_df_10.to_json(orient="split")
        expected_samples_df_10 = expected_results[test_name]["samples_df_10"] = samples_df_10.to_json(orient="split")
        expected_file_names_10 = expected_results[test_name]["file_names_10"] = file_names_10

        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_10.to_json(orient="split") == expected_results_df_10)
    assert(samples_df_10.to_json(orient="split") == expected_samples_df_10)
    assert(file_names_10 == expected_file_names_10)

    #test given invalid validation
    results_df_11, samples_df_11, file_names_11 = parse_files(
        input_files,
        default_settings["experiments"],
        default_settings["progress_points"],
        [],
        0,
        ["fast", "fast", "12", "1024","0"],
        default_settings["recursive"],
        default_settings["cli"],
    )

    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    test_name  = "test_parse_files_11"
    assert(test_name in expected_results.keys())
    expected_results_df_11 = expected_results[test_name]["results_df_11"]
    expected_samples_df_11 = expected_results[test_name]["samples_df_11"]
    expected_file_names_11 = expected_results[test_name]["file_names_11"]

    # else:
    #     expected_results[test_name]={}
    #     expected_results_df_11 = expected_results[test_name]["results_df_11"] = results_df_11.to_json(orient="split")
    #     expected_samples_df_11 = expected_results[test_name]["samples_df_11"] = samples_df_11.to_json(orient="split")
    #     expected_file_names_11 = expected_results[test_name]["file_names_11"] = file_names_11

        # with open ("test_results.json", "w") as test_results:
        #     json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(results_df_11.to_json(orient="split") == expected_results_df_11)
    assert(samples_df_11.to_json(orient="split") == expected_samples_df_11)
    assert(file_names_11 == expected_file_names_11)

    #test given invalid validation len 
    with pytest.raises(Exception) as e_info:
        parse_files(
            input_files,
            default_settings["experiments"],
            default_settings["progress_points"],
            [],
            0,
            ["fast", "fast", "12", "1024","0", "10"],
            default_settings["recursive"],
            default_settings["cli"],
        )

def test_find_causal_files():
    # given a valid directory
    find_causal_files([workload_dir], default_settings["verbose"], False)
    assert(len(find_causal_files([workload_dir], default_settings["verbose"],False)) == 4)
    assert(find_causal_files([workload_file], default_settings["verbose"], False) == [workload_file])

    # given invalid directory 
    with pytest.raises(Exception) as e_info:
        find_causal_files(["nonsense"], default_settings["verbose"], False)

    # given valid directory with recursive
    find_causal_files([workload_dir], default_settings["verbose"], True)
    assert(len(find_causal_files([workload_dir], default_settings["verbose"],True)) == 6)
    assert(find_causal_files([workload_file], default_settings["verbose"], True) == [workload_file])

    # given invalid directory with recursive
    with pytest.raises(Exception) as e_info:
        find_causal_files(["nonsense"], default_settings["verbose"], True)

def test_set_num_stddev():
    assert(True)

def test_process_data():
    # test with valid data
    with open(workload_file) as file:
        _data = json.loads(file.read())
        dict_data ={}
        data = process_data(dict_data, _data, ".*", ".*")
        assert (list(dict_data.keys()) == 
                ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])
        assert (list(data.keys()) == 
                ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])

        data = process_data({}, _data, ".*", "fast")
        assert (list(dict_data.keys()) == 
                ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])
        assert (list(data.keys()) == [])
        
        data = process_data({}, _data, "fast", ".*")
        assert (list(dict_data.keys()) == 
                ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])
        assert (list(data.keys()) == ['cpu_fast_func(long, int)'])

        data = process_data({}, _data, ".*", "impl")
        assert (list(dict_data.keys()) == 
                ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])
        assert (list(data.keys()) == ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])

        
        data = process_data({}, _data, "impl", ".*")
        assert (list(dict_data.keys()) == 
                ['cpu_fast_func(long, int)', 'cpu_slow_func(long, int)', 'bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])
        assert (list(data.keys()) == ['bool rng_func_impl<false>(long, unsigned long)', 'bool rng_func_impl<true>(long, unsigned long)'])

    assert(True)

def test_compute_speedups():
    with open(workload_file) as file:
        _data = json.loads(file.read())
        
        dict_data={}
        dict_data[workload_file] = process_data({}, _data, ".*", ".*")
        
        # Testing verbosity
        results_df = compute_speedups(dict_data, [], default_settings["min_points"], [], 3)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups"
            if test_name in expected_results:
                expected_results_df = expected_results[test_name]["results_df"]

            else:
                expected_results[test_name]={}
                expected_results_df = expected_results[test_name]["results_df"] = results_df.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df.to_json(orient="split") == expected_results_df)
        
        # Testing verbosity
        results_df_1 = compute_speedups(dict_data, [], default_settings["min_points"], [], 2)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_1"
            if test_name in expected_results:
                expected_results_df_1 = expected_results[test_name]["results_df_1"]

            else:
                expected_results[test_name]={}
                expected_results_df_1 = expected_results[test_name]["results_df_1"] = results_df_1.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_1.to_json(orient="split") == expected_results_df_1)

        # Testing verbosity
        results_df_2 = compute_speedups(dict_data, [], default_settings["min_points"], [], 1)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_2"
            if test_name in expected_results:
                expected_results_df_2 = expected_results[test_name]["results_df_2"]

            else:
                expected_results[test_name]={}
                expected_results_df_2 = expected_results[test_name]["results_df_2"] = results_df_2.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_2.to_json(orient="split") == expected_results_df_2)

        # Testing verbosity
        results_df_3 = compute_speedups(dict_data, [], default_settings["min_points"], [], 0)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_3"
            if test_name in expected_results:
                expected_results_df_3 = expected_results[test_name]["results_df_3"]

            else:
                expected_results[test_name]={}
                expected_results_df_3 = expected_results[test_name]["results_df_3"] = results_df_3.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_3.to_json(orient="split") == expected_results_df_3)

        # Testing verbosity
        results_df_4 = compute_speedups(dict_data, [], default_settings["min_points"], [], 4)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_4"
            if test_name in expected_results:
                expected_results_df_4 = expected_results[test_name]["results_df_4"]

            else:
                expected_results[test_name]={}
                expected_results_df_4 = expected_results[test_name]["results_df_4"] = results_df_4.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_4.to_json(orient="split") == expected_results_df_4)

        # min points too high
        results_df_5 = compute_speedups(dict_data, [], 247, [], 3)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_5"
            if test_name in expected_results:
                expected_results_df_5 = expected_results[test_name]["results_df_5"]

            else:
                expected_results[test_name]={}
                expected_results_df_5 = expected_results[test_name]["results_df_5"] = results_df_5.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_5.to_json(orient="split") == expected_results_df_5)
        
        # min points 0
        results_df_6 = compute_speedups(dict_data, [], 0, [], 3)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_6"
            if test_name in expected_results:
                expected_results_df_6 = expected_results[test_name]["results_df_6"]

            else:
                expected_results[test_name]={}
                expected_results_df_6 = expected_results[test_name]["results_df_6"] = results_df_6.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_6.to_json(orient="split") == expected_results_df_6)
        
        # min points 1
        results_df_7 = compute_speedups(dict_data, [], 1, [], 3)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_7"
            if test_name in expected_results:
                expected_results_df_7 = expected_results[test_name]["results_df_7"]

            else:
                expected_results[test_name]={}
                expected_results_df_7 = expected_results[test_name]["results_df_7"] = results_df_7.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_7.to_json(orient="split") == expected_results_df_7)

        # empty dict_data
        results_df_8 = compute_speedups({}, [], 0, [], 3)
        with open ("test_results.json", "r") as test_results:
            expected_results = json.load(test_results)
            test_name  = "test_compute_speedups_8"
            if test_name in expected_results:
                expected_results_df_8 = expected_results[test_name]["results_df_8"]

            else:
                expected_results[test_name]={}
                expected_results_df_8 = expected_results[test_name]["results_df_8"] = results_df_8.to_json(orient="split")
                with open ("test_results.json", "w") as test_results:
                    json.dump(expected_results, test_results, sort_keys=True, indent=4)
            assert(results_df_8.to_json(orient="split") == expected_results_df_8)

    
    assert(True)

def test_get_validations():
    assert(True)

def test_compute_sorts():
    assert(True)

def test_parse_uploaded_file():
    assert True

def test_get_data_point():
    assert True

def get_speedup_data():
    assert True

def set_up(ip_addr = "localhost", ip_port = "8051"):
    # works for linux, no browser pops up
    fireFoxOptions = webdriver.FirefoxOptions()
    fireFoxOptions.add_argument("--headless")
    driver = webdriver.Firefox(options=fireFoxOptions)
    driver.get("http://"+ip_addr+":"+ip_port+"/")
    # end works for linux

    return driver


# @pytest.fixture(autouse=True)
# def capfd(self, capfd):
#     self.capfd = capfd


# test order of chart titles
def test_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()
    time.sleep(10)

    driver = set_up()
    #time.sleep(10)
    main_page = page.MainPage(driver)

    expected_title_set = []
    captured_output = main_page.get_titles()
    t.terminate()
    t.join()
    driver.quit()    

    test_name = "test_title_order"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if "test_title_order" in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results["test_title_order"] = {}
        expected_title_set = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert (captured_output == expected_title_set)


def test_alphabetical_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()
    time.sleep(10)
    driver = set_up()
    main_page = page.MainPage(driver)

    expected_output = []
    captured_output = main_page.get_alphabetical_titles()
    captured_histogram_data = main_page.get_histogram_data()
    captured_plot_data = main_page.get_plot_data()

    t.terminate()
    t.join()
    driver.quit()    
    
    test_name = "test_alphabetical_title_order"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_output = expected_results[test_name]["titles"]
    else:
        expected_results[test_name] = {}
        expected_output = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert ( captured_output == expected_output)


def test_max_speedup_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()
    time.sleep(10)
    driver = set_up()
    
    main_page = page.MainPage(driver)
    captured_output = main_page.get_max_speedup_titles()
    captured_histogram_data = main_page.get_histogram_data()
    captured_plot_data = main_page.get_plot_data()
    expected_title_set = []

    t.terminate()
    t.join()
    driver.quit()
    test_name = "test_max_speedup_title_order"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name]={}
        expected_title_set = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert ( captured_output == expected_title_set)


def test_min_speedup_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()
    time.sleep(10)
    driver = set_up()
    
    main_page = page.MainPage(driver)

    expected_title_set = []
    captured_output = main_page.get_min_speedup_titles()
    captured_histogram_data = main_page.get_histogram_data()
    captured_plot_data = main_page.get_plot_data()

    t.terminate()
    t.join()
    driver.quit()
    test_name = "test_min_speedup_title_order"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name] ={} 
        expected_title_set = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert( captured_output == expected_title_set)


def test_impact_title_order():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()

    time.sleep(10)
    driver = set_up()
    
    main_page = page.MainPage(driver)

    expected_title_set = []
    captured_output = main_page.get_impact_titles()
    captured_histogram_data = main_page.get_histogram_data()
    captured_plot_data = main_page.get_plot_data()

    t.terminate()
    t.join()
    driver.quit()
    test_name = "test_impact_title_order"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name] ={}
        expected_title_set = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert( captured_output == expected_title_set)


def test_min_points_slider():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()
    time.sleep(10)

    driver = set_up()
    # driver.refresh()
    main_page = page.MainPage(driver)
    expected_title_set = []
    captured_output = main_page.get_min_points_titles()
    captured_histogram_data = main_page.get_histogram_data()
    captured_plot_data = main_page.get_plot_data()

    t.terminate()
    t.join()
    driver.quit()
    test_name = "test_min_points_slider"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name]={}
        expected_title_set = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert( captured_output == expected_title_set)

def test_workload_flag_gui():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    t = multiprocessing.Process(target=causal, args=(parser_args,))
    t.start()
    time.sleep(10)
    driver = set_up()
    
    main_page = page.MainPage(driver)
    expected_title_set = []
    captured_output = main_page.get_titles()

    t.terminate()
    t.join()
    driver.quit()
    test_name = "test_workload_flag_gui"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name] ={}
        expected_title_set = expected_results[test_name]["titles"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert( captured_output == expected_title_set)


def test_verbose_gui_flag_1():
    t = subprocess.Popen([sys.executable, "-m","source", "-w", workload_dir,"--verbose","1", "-n", "0"], stdout=subprocess.PIPE)

    time.sleep(10)
    driver = set_up()
    main_page = page.MainPage(driver)

    expected_title_set = []
    expected_output = ""
    captured_title_set = main_page.get_titles()
    #driver.close()
    t.terminate()
    #t.join()
    driver.quit()    
    captured_output = t.communicate(timeout=15)[0].decode('utf-8')

    test_name = "test_verbose_gui_flag_1"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_output = expected_results[test_name]["cli"]
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name]={}

        expected_output = expected_results[test_name]["cli"] = captured_output
        expected_title_set = expected_results[test_name]["titles"] = captured_title_set
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(captured_title_set == expected_title_set)
    assert(captured_output == expected_output) 

 # works with cli not gui
    # t = subprocess.run(["omnitrace-causal-plot", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--verbose","2", "-n", "0", "--cli"], capture_output=True)
        # out, err = self.capfd.readouterr()

#output = subprocess.check_output( stdin=t.stdout)
def test_verbose_gui_flag_2():
    my_parser = create_parser(default_settings)
    parser_args = my_parser.parse_args(
        [
            "-w",
            workload_dir,
        ]
    )
    

    # t = multiprocessing.Process(target=causal, args=(parser_args,))
    # t.start()
    #print("opening")
    # works with cli not gui
    # t = subprocess.run(["omnitrace-causal-plot", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--verbose","2", "-n", "0", "--cli"], capture_output=True)

    t = subprocess.Popen([sys.executable, "-m","source", "-w", workload_dir,"--verbose","2", "-n", "0"], stdout=subprocess.PIPE)

    expected_title_set = []
    #print("\nexpected_title_set: ", expected_title_set)
    time.sleep(10)
    driver = set_up()
    
    # driver.refresh()
    #time.sleep(20)
    main_page = page.MainPage(driver)

    
    # out, err = self.capfd.readouterr()

    captured_title_set = main_page.get_titles()
    
    #print("\nexpected_title_set_run: ", captured_title_set)
    #driver.close()
    #output = subprocess.check_output( stdin=t.stdout)
    t.terminate()
    #t.join()
    driver.quit()    
    captured_output = t.communicate(timeout=15)[0].decode('utf-8')
    #print(captured_output)
    # t.terminate()
    #t.join()
    expected_cli_output=""

    test_name = "test_verbose_gui_flag_2"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        
        expected_title_set = expected_results[test_name]["titles"]
        expected_cli_output = expected_results[test_name]["cli"]
    else:
        expected_results[test_name]={}
        expected_title_set = expected_results[test_name]["titles"] = captured_title_set
        expected_cli_output = expected_results[test_name]["cli"] = captured_output
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)
    
    assert( captured_output == expected_cli_output )
    assert( captured_title_set == expected_title_set )


def test_verbose_gui_flag_3():
    #t = subprocess.Popen(["omnitrace-causal-plot", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--verbose","3", "-n", "0"], stdout=subprocess.PIPE)
    t = subprocess.Popen([sys.executable, "-m","source", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--verbose","3", "-n", "0"], stdout=subprocess.PIPE)

    time.sleep(10)
    driver = set_up()

    ## driver.refresh()
    #time.sleep(20)
    main_page = page.MainPage(driver)

    expected_title_set = []

    # out, err = self.capfd.readouterr()
    expected_output = ""

    captured_title_set = main_page.get_titles()
    print("\nexpected_title_set: ", expected_title_set)
    #driver.close()
    #output = subprocess.check_output( stdin=t.stdout)
    t.terminate()
    #t.join()
    driver.quit()    
    captured_output = t.communicate(timeout=15)[0].decode('utf-8')

    #print(captured_output)
    test_name = "test_verbose_gui_flag_3"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_output = expected_results[test_name]["cli"]
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name]={}
        expected_output = expected_results[test_name]["cli"] = captured_output
        expected_title_set = expected_results[test_name]["titles"] = captured_title_set
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(captured_title_set == expected_title_set)
    assert(captured_output == expected_output) 

def test_ip_addr_flag():
    t = subprocess.Popen([sys.executable, "-m","source", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--ip_addr","0.0.0.1"], stdout=subprocess.PIPE)

    time.sleep(10)
    driver = set_up(ip_addr="0.0.0.1")
    main_page = page.MainPage(driver)

    expected_title_set = []
    # out, err = self.capfd.readouterr()
    expected_output = ""
    captured_title_set = main_page.get_titles()
    #print("\nexpected_title_set: ", expected_title_set)
    #driver.close()
    #output = subprocess.check_output( stdin=t.stdout)
    t.terminate()
    #t.join()
    driver.quit()   
    captured_output = t.communicate(timeout=15)[0].decode('utf-8')
    #print(capture)

    test_name = "test_ip_addr_flag"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        
        expected_output = expected_results[test_name]["cli"]
        captured_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name]={}
        expected_output = expected_results[test_name]["cli"] = captured_output
        expected_results[test_name]["titles"] = captured_title_set
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(captured_title_set == expected_title_set)
    assert(captured_output == expected_output) 
        
def test_ip_port_flag():
    t = subprocess.Popen([sys.executable, "-m","source", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--port","8052"], stdout=subprocess.PIPE)

    time.sleep(10)
    driver = set_up(ip_port="8052")
    main_page = page.MainPage(driver)

    expected_title_set = []
    expected_output = "8052"

    captured_title_set = main_page.get_titles()
    t.terminate()
    driver.quit()    
    captured_output = t.communicate(timeout=15)[0].decode('utf-8')

    test_name = "test_ip_port_flag"
    with open ("test_results.json", "r") as test_results:
        expected_results = json.load(test_results)
    if test_name in expected_results:
        expected_title_set = expected_results[test_name]["titles"]
    else:
        expected_results[test_name]={}
        expected_results[test_name]["titles"] = captured_title_set
        with open ("test_results.json", "w") as test_results:
            json.dump(expected_results, test_results, sort_keys=True, indent=4)

    assert(captured_title_set == expected_title_set)
    assert(expected_output in captured_output) 

def test_experiments_flag():
    return true
    t = subprocess.Popen([sys.executable, "-m","source", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--verbose","3", "-n", "0", "-e", ".*"], stdout=subprocess.PIPE)

    time.sleep(20)
    driver = set_up()

    ## driver.refresh()
    #time.sleep(20)
    main_page = page.MainPage(driver)

    expected_title_set = [
        "Selected Causal Profiles",
        "cpu_slow_func(long, int)",
        "/home/jose/omnitrace/examples/causal/causal.cpp:165",
        "cpu_fast_func(long, int)",
    ]
    # out, err = self.capfd.readouterr()
    expected_output = ""

    expected_title_set_run = main_page.get_titles()
    print("\nexpected_title_set: ", expected_title_set)
    driver.close()
    #output = subprocess.check_output( stdin=t.stdout)
    t.terminate()
    t.join()
    driver.quit()    
    captured_output = t.communicate(timeout=15)

    print(captured_output)
    with open("capture_output.txt", "w") as text_file:
        text_file.write(captured_output[0].decode('utf-8'))
    assert(expected_title_set_run == expected_title_set)
    assert(expected_output in captured_output)

def test_progress_points_flag(capfd):
    return true
    t = subprocess.Popen([sys.executable, "-m","source", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/", "-n", "0", "--cli", "-p", ".*"], stdout=subprocess.PIPE)
    #t = subprocess.run(["omnitrace-causal-plot", "-w","/home/jose/omnitrace/source/python/gui/workloads/omnitrace-tests-output/causal-cpu-omni-fast-func-e2e/causal/","--verbose","2", "-n", "0", "--cli", "-p", ".*"], capture_output=True)
    time.sleep(20)
    driver = set_up()

    ## driver.refresh()
    #time.sleep(20)
    main_page = page.MainPage(driver)

    expected_title_set = [
        "Selected Causal Profiles",
        "cpu_slow_func(long, int)",
        "/home/jose/omnitrace/examples/causal/causal.cpp:165",
        "cpu_fast_func(long, int)",
    ]
    # out, err = self.capfd.readouterr()
    expected_output = ""

    #expected_title_set_run = main_page.get_titles()
    print("\nexpected_title_set: ", expected_title_set)
    driver.close()
    #output = subprocess.check_output( stdin=t.stdout)
    captured_output, err = capfd.readouterr()

    print(captured_output)
    with open("capture_output.txt", "w") as text_file:
        text_file.write(captured_output)
    #assert(expected_title_set_run == expected_title_set)
    assert(expected_output in captured_output) 

    # def test_num_points_flag():
    #     self.assertTrue(True,True)

    # def test_speedups_flag():
    #     self.assertTrue(True,True)

    # def test_std_dev_flag():
    #     self.assertTrue(True,True)

    # def test_validate_flag():
    #     self.assertTrue(True,True)

# if __name__ == "__main__":
#     unittest.main()
