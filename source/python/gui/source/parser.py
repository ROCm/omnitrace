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
import json
import jsondiff
import re
import pandas as pd
import numpy as np
import math
from collections import OrderedDict
import os

num_stddev = 1


def mean(_data):
    return sum(_data) / float(len(_data)) if len(_data) > 0 else 0.0


def stddev(_data):
    if len(_data) == 0:
        return 0.0
    _mean = mean(_data)
    _variance = sum([((x - _mean) ** 2) for x in _data]) / float(len(_data))
    return _variance**0.5


class validation(object):
    def __init__(self, _exp_re, _pp_re, _virt, _expected, _tolerance):
        self.experiment_filter = re.compile(_exp_re)
        self.progress_pt_filter = re.compile(_pp_re)
        self.virtual_speedup = int(_virt)
        self.program_speedup = float(_expected)
        self.tolerance = float(_tolerance)

    def validate(
        self,
        _exp_name,
        _pp_name,
        _virt_speedup,
        _prog_speedup,
        _prog_speedup_stddev,
        _base_speedup_stddev,
    ):
        if (
            not re.search(self.experiment_filter, _exp_name)
            or not re.search(self.progress_pt_filter, _pp_name)
            or _virt_speedup != self.virtual_speedup
        ):
            return None
        _tolerance = self.tolerance
        if _base_speedup_stddev > 2.0 * self.tolerance:
            sys.stderr.write(
                f"  [{_exp_name}][{_pp_name}][{_virt_speedup}] base speedup has stddev > 2 * tolerance (+/- {_base_speedup_stddev:.3f}). Relaxing validation...\n"
            )
            _tolerance += math.sqrt(_base_speedup_stddev)
        elif _prog_speedup_stddev > 2.0 * self.tolerance:
            sys.stderr.write(
                f"  [{_exp_name}][{_pp_name}][{_virt_speedup}] program speedup has stddev > 2 * tolerance (+/- {_prog_speedup_stddev:.3f}). Relaxing validation...\n"
            )
            _tolerance += math.sqrt(_prog_speedup_stddev)
        return _prog_speedup >= (self.program_speedup - _tolerance) and _prog_speedup <= (
            self.program_speedup + _tolerance
        )


class throughput_point(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.delta = []
        self.duration = []

    def __iadd__(self, _data):
        self.delta += [float(_data[0])]
        self.duration += [float(_data[1])]

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def get_data(self):
        return [x / y for x, y in zip(self.duration, self.delta)]

    def mean(self):
        return sum(self.duration) / sum(self.delta)


class latency_point(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.arrivals = []
        self.departures = []
        self.duration = []

    def __iadd__(self, _data):
        self.arrivals += [float(_data[0])]
        self.departures += [float(_data[1])]
        self.duration += [float(_data[2])]

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def get_data(self):
        _duration = sum(self.duration)
        return [x / _duration for x in self.duration]

    def mean(self):
        rate = sum(self.arrivals) / sum(self.duration)
        return sum(self.get_data()) / rate

    def __init__(self, _data):
        self.data = _data

    def get_impact(self):
        """
        speedup_c = [x.compute_speedup() for x in self.data]
        speedup_v = [x.virtual_speedup() for x in self.data]
        impact = []
        for i in range(len(self.data) - 1):
            x = speedup_v[i + 1] - speedup_v[i]
            y_low = speedup_c[i]
            y_upp = speedup_c[i + 1]
            a_low = x * min([y_low, y_upp])
            a_high = 0.5 * x * (max([y_low, y_upp]) - min([y_low, y_upp]))
            impact += [a_low + a_high]
        """
        impact = [x.compute_speedup() for x in self.data]
        return [sum(impact), mean(impact), stddev(impact)]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        _impact_v = self.get_impact()
        _name = self.data[0].get_name()
        _prog = self.data[0].prog
        _impact = [
            f"[{_name}][{_prog}][sum]  impact: {_impact_v[0]:6.1f}",
            f"[{_name}][{_prog}][avg]  impact: {_impact_v[1]:6.1f} +/- {_impact_v[2]:6.2f}",
        ]
        return "\n".join([f"{x}" for x in self.data] + _impact)

    def __lt__(self, rhs):
        self.data.sort()
        return self.get_impact()[0] < rhs.get_impact()[0]


class line_speedup(object):
    def __init__(self, _name="", _prog="", _exp_data=None, _exp_base=None):
        self.name = _name
        self.prog = _prog
        self.data = _exp_data
        self.base = _exp_base

    def virtual_speedup(self):
        if self.data is None or self.base is None:
            return 0.0
        return self.data.speedup

    def compute_speedup(self):
        if self.data is None or self.base is None:
            return 0.0
        return ((self.base.mean() - self.data.mean()) / self.base.mean()) * 100

    def compute_speedup_stddev(self):
        if self.data is None or self.base is None:
            return 0.0
        _data = []
        _base = self.base.mean()
        for ditr in self.data.get_data():
            _data += [((_base - ditr) / _base) * 100]
        return stddev(_data)

    def get_name(self):
        return ":".join(
            [
                os.path.basename(x) if os.path.isfile(x) else x
                for x in self.name.split(":")
            ]
        )

    def __str__(self):
        if self.data is None or self.base is None:
            return f"{self.name}"
        _line_speedup = self.compute_speedup()
        _line_stddev = (
            float(num_stddev) * self.compute_speedup_stddev()
        )  # 3 stddev == 99.87%
        _name = self.get_name()
        return f"[{_name}][{self.prog}][{self.data.speedup:3}] speedup: {_line_speedup:6.1f} +/- {_line_stddev:6.2f} %"

    def __eq__(self, rhs):
        return (
            self.name == rhs.name
            and self.prog == rhs.prog
            and self.data == rhs.data
            and self.base == rhs.base
        )

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        if self.name != rhs.name:
            return self.name < rhs.name
        elif self.prog != rhs.prog:
            return self.prog < rhs.prog
        elif self.data != rhs.data:
            return self.data < rhs.data
        elif self.base != rhs.base:
            return self.base < rhs.base
        return False


class experiment_progress(object):
    def __init__(self, _data):
        self.data = _data

    def get_impact(self):
        speedup_c = [x.compute_speedup() for x in self.data]
        speedup_v = [x.virtual_speedup() for x in self.data]
        impact = []
        for i in range(len(self.data) - 1):
            x = speedup_v[i + 1] - speedup_v[i]
            y = [speedup_c[i], speedup_c[i + 1]]
            y_min = min(y)
            y_max = max(y)
            a_low = x * y_min
            a_upp = 0.5 * x * (y_max - y_min)
            impact += [a_low + a_upp]
        # impact = [x.compute_speedup() for x in self.data]
        return [sum(impact), mean(impact), stddev(impact)]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        _impact_v = self.get_impact()
        _name = self.data[0].get_name()
        _prog = self.data[0].prog
        _impact = [
            f"[{_name}][{_prog}][sum]  impact: {_impact_v[0]:6.1f}",
            f"[{_name}][{_prog}][avg]  impact: {_impact_v[1]:6.1f} +/- {_impact_v[2]:6.2f}",
        ]
        return "\n".join([f"{x}" for x in self.data] + _impact)

    def __lt__(self, rhs):
        self.data.sort()
        return self.get_impact()[0] < rhs.get_impact()[0]


def process_samples(data, _data):
    if not _data:
        return data
    for record in _data["omnitrace"]["causal"]["records"]:
        for samp in record["samples"]:
            _info = samp["info"]
            _count = samp["count"]
            _func = _info["dfunc"]
            if _func not in data:
                data[_func] = 0
            data[_func] += _count
            for dwarf_entry in _info["dwarf_info"]:
                _name = "{}:{}".format(dwarf_entry["file"], dwarf_entry["line"])
                if _name not in data:
                    data[_name] = 0
                data[_name] += _count
    return data


def process_data(data, _data, experiments, progress_points):
    def find_or_insert(_data, _value, _type):
        if _value not in _data:
            if _type == "throughput":
                _data[_value] = throughput_point(_value)
            elif _type == "latency":
                _data[_value] = latency_point(_value)
        return _data[_value]

    if not _data:
        return data
    _selection_filter = re.compile(experiments)
    _progresspt_filter = re.compile(progress_points)
    for record in _data["omnitrace"]["causal"]["records"]:
        for exp in record["experiments"]:
            _speedup = exp["virtual_speedup"]
            _duration = exp["duration"]
            _file = exp["selection"]["info"]["file"]
            _line = exp["selection"]["info"]["line"]
            _func = exp["selection"]["info"]["dfunc"]
            _sym_addr = exp["selection"]["symbol_address"]
            _selected = ":".join([_file, f"{_line}"]) if _sym_addr == 0 else _func
            if not re.search(_selection_filter, _selected):
                continue
            if _selected not in data:
                data[_selected] = {}
            for pts in exp["progress_points"]:
                _name = pts["name"]
                if not re.search(_progresspt_filter, _name):
                    continue
                if _name not in data[_selected]:
                    data[_selected][_name] = {}
                if "delta" in pts:
                    _delt = pts["delta"]
                    if _delt > 0:
                        itr = find_or_insert(
                            data[_selected][_name], _speedup, "throughput"
                        )
                        itr += [_delt, _duration]
                    elif "arrivals" in pts:
                        if pts["arrivals"] > 0:
                            itr = find_or_insert(
                                data[_selected][_name], _speedup, "latency"
                            )
                            itr += [pts["arrival"], pts["departure"], _duration]
                else:
                    _delt = pts["laps"]
                    if _delt > 0:
                        itr = find_or_insert(data[_selected][_name], _speedup)
                        itr += [_delt, _duration]
    return data


def compute_speedups(_data, speedups=[], num_points=0, validate=[], CLI=False):
    out = pd.DataFrame()
    data = {}
    for selected, pitr in _data.items():
        if selected not in data:
            data[selected] = {}
        for progpt, ditr in pitr.items():
            data[selected][progpt] = OrderedDict(sorted(ditr.items()))
    from os.path import dirname

    ret = []
    for selected, pitr in _data.items():
        for progpt, ditr in pitr.items():
            if 0 not in ditr.keys():
                # print(f"missing baseline data for {progpt} in {selected}...")
                continue
            _baseline = ditr[0].mean()
            for speedup, itr in ditr.items():
                if len(speedups) > 0 and speedup not in speedups:
                    continue
                if speedup != itr.speedup:
                    raise ValueError(f"in {selected}: {speedup} != {itr.speedup}")
                _val = line_speedup(selected, progpt, itr, ditr[0])
                ret.append(_val)
    ret.sort()
    _last_name = None
    _last_prog = None
    result = []
    for itr in ret:
        if itr.name != _last_name or itr.prog != _last_prog:
            result.append([])
        result[-1].append(itr)
        _last_name = itr.name
        _last_prog = itr.prog
        _data = []
    for itr in result:
        experiment_prog = experiment_progress(itr)
        _data.append(experiment_prog)
        if len(itr) != 0:
            impact = experiment_prog.get_impact()
            for itrx in itr:
                speedup = itrx.compute_speedup()
                if speedup <= 200 and speedup >= -100:
                    out = pd.concat(
                        [
                            out,
                            pd.DataFrame(
                                {
                                    "idx": [(itrx.prog, itrx.name)],
                                    "progress points": [itrx.prog],
                                    "point": [itrx.name],
                                    "Line Speedup": [itrx.virtual_speedup()],
                                    "Program Speedup": [speedup],
                                    "impact sum": impact[0],
                                    "impact avg": impact[1],
                                    "impact err": float(impact[2]),
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
    _data.sort()

    if CLI:
        for itr in _data:
            if len(itr) < num_points:
                continue
            print("")
            print(f"{itr}")
    sys.stdout.flush()
    validations = get_validations(validate)
    expected_validations = len(validations)
    correct_validations = 0
    if expected_validations > 0:
        print(f"\nPerforming {expected_validations} validations...\n")
        for eitr in _data:
            _experiment = eitr.data[0].get_name()
            _progresspt = eitr.data[0].prog
            _base_speedup_stddev = eitr.data[0].compute_speedup_stddev()
            for ditr in eitr.data:
                _virt_speedup = ditr.virtual_speedup()
                _prog_speedup = ditr.compute_speedup()
                _prog_speedup_stddev = ditr.compute_speedup_stddev()
                for vitr in validations:
                    _v = vitr.validate(
                        _experiment,
                        _progresspt,
                        _virt_speedup,
                        _prog_speedup,
                        _prog_speedup_stddev,
                        _base_speedup_stddev,
                    )
                    if _v is None:
                        continue
                    if _v is True:
                        correct_validations += 1
                    else:
                        sys.stderr.write(
                            f"  [{_experiment}][{_progresspt}][{_virt_speedup}] failed validation: {_prog_speedup:8.3f} != {vitr.program_speedup} +/- {vitr.tolerance}\n"
                        )
    if expected_validations != correct_validations:
        sys.stderr.flush()
        sys.stderr.write(
            f"\nCausal profiling predictions not validated. Expected {expected_validations}, found {correct_validations}\n"
        )
        sys.stderr.flush()
        sys.exit(-1)
    elif expected_validations > 0:
        print(f"Causal profiling predictions validated: {expected_validations}")

    return out


def get_validations(validate):
    data = []
    _len = len(validate)
    if _len == 0:
        return data
    elif _len % 5 != 0:
        raise ValueError(
            "validation requires format: {experiment regex} {progress-point regex} {virtual-speedup} {expected-speedup} {tolerance} (i.e. 5 args per validation. There are {} extra/missing arguments".format(
                _len % 5
            )
        )
    v = validate
    for i in range(int(_len / 5)):
        off = 5 * i
        data.append(
            validation(v[off + 0], v[off + 1], v[off + 2], v[off + 3], v[off + 4])
        )
    return data


def compute_sorts(_data):
    Max_speedup_order = _data.sort_values(
        by="Program Speedup", ascending=False
    ).point.unique()
    Min_speedup_order = _data.sort_values(
        by="Program Speedup", ascending=True
    ).point.unique()

    # just a list of experiments
    # impactOrder = pd.DataFrame(_data["point"].unique(), columns=["point"])
    point_counts = _data.point.value_counts()
    speedups = pd.DataFrame(_data["Line Speedup"].unique(), columns=["Line Speedup"])
    # for imp_idx, curr in impactOrder.iterrows():
    #    prev = pd.Series(dtype="float64")
    #    prev_speedup = 0
    #    progress_point = curr[0]
    #    data_subset = _data[_data["point"] == progress_point]
    #    area = 0
    #    max_norm_area = 0

    # subset is by progress point...
    # for index_sub, data_point in data_subset.iterrows():

    #    for speedup in speedups["Line Speedup"]:
    #        data_point = data_subset[data_subset["Line Speedup"] == speedup]
    #        if speedup == 0 or data_point.empty:
    #            continue
    #        if prev.empty:
    #            prev = data_point
    #            prev_speedup = speedup
    #        else:
    #            avg_progress_speedup = (
    #                prev["Program Speedup"].sum() + data_point["Program Speedup"].sum()
    #            ) / (prev["Program Speedup"].size + data_point["Program Speedup"].size)
    #            area = area + avg_progress_speedup * (
    #                speedup - prev_speedup
    #            )
    #            norm_area = area / speedup
    #            if norm_area > max_norm_area:
    #                max_norm_area = norm_area
    #            prev = data_point
    #    impactOrder.at[imp_idx, "area"] = max_norm_area
    # impactOrder = impactOrder.sort_values(by="area", ascending=False)
    # impactOrder = impactOrder["point"].unique()
    _data["Max Speedup"] = np.nan
    _data["Min Speedup"] = np.nan
    # _data["impact"] = np.nan
    _data["point count"] = np.nan
    for index in _data.index:
        _data.at[index, "Max Speedup"] = np.where(
            Max_speedup_order == _data.at[index, "point"]
        )[0][0]
        # _data.at[index, "impact"] = np.where(impactOrder == _data.at[index, "point"])[0]
        _data.at[index, "Min Speedup"] = np.where(
            Min_speedup_order == _data.at[index, "point"]
        )[0][0]
        _data.at[index, "point count"] = point_counts[_data.at[index, "point"]]
    return _data


def isValidDataPoint(data):
    return math.isnan(data) == False and math.isinf(data) == False


def getValue(splice):
    if "type" in splice:
        # Something might happen here later
        print("type found in coz file, finish adding code")
        sys.exit(1)
    if "difference" in splice or "speedup" in splice:
        return float(splice[1])
    else:
        return splice[1]


def addThroughput(df, experiment, value):
    # maybe id is issue
    ix = (experiment["selected"], value["name"], experiment["speedup"])
    df_points = list(df.index)
    if ix not in df_points:
        new_row = pd.DataFrame(
            data={
                # "selected" : [experiment["selected"]],
                # "speedup" : [float(experiment["speedup"])],
                "delta": [float(value["delta"])],
                "duration": [float(experiment["duration"])],
                "type": ["throughput"],
            },
            index=[ix],
        )
        df = pd.concat([new_row, df])
    else:
        df["delta"][ix] = df["delta"][ix] + float(value["delta"])
        df["duration"][ix] = df["duration"][ix] + float(experiment["duration"])
    return df


def addLatency(df, experiment, value):
    ix = (experiment["selected"], experiment["speedup"])
    df_points = list(df.index)
    if ix not in df_points:
        new_row = pd.Dataframe(
            data={
                "selected": [experiment["selected"]],
                "point": [value["name"]],
                "speedup": [float(experiment["speedup"])],
                "arrivals": [float(value["arrivals"])],
                "departures": [float(value["departures"])],
                "duration": [0],
                "type": ["latency"],
            }
        )

    if value.duration == 0:
        df["difference"] = value.difference
    else:
        duration = df["duration"][ix] + float(experiment["duration"])
        df["difference"][ix] = df["difference"][ix] * df["duration"][ix] / duration
        df["difference"][ix] = (
            df["difference"]
            + (float(value["difference"]) * float(experiment["duration"])) / duration
        )

    df["duration"] = df["duration"] + float(experiment["duration"])

    return df


def parseFiles(files, experiments=".*", progress_points=".*", speedups=[], CLI=False):
    data = pd.DataFrame()
    out = pd.DataFrame()
    samples = {}
    dict_data = {}
    cli_out = []

    name_wo_ext = lambda x: x.replace(".json", "").replace(".coz", "")

    json_files = [x for x in filter(lambda y: y.endswith(".json"), files)]
    coz_files = [x for x in filter(lambda y: y.endswith(".coz"), files)]
    base_files = [name_wo_ext(x) for x in coz_files + json_files]
    read_files = []

    # prefer JSON files first
    files = json_files + coz_files
    for file in files:
        _base_name = name_wo_ext(file)
        # do not read in a COZ file if the JSON already read
        if _base_name in read_files:
            continue

        if file.endswith(".json"):
            with open(file, "r") as j:
                _data = json.load(j)
                # make sure the JSON is an omnitrace causal JSON
                if "omnitrace" not in _data or "causal" not in _data["omnitrace"]:
                    continue
                dict_data = process_data(dict_data, _data, experiments, progress_points)
                sample_data = process_samples(sample_data, _data)

                dict_data = compute_speedups(dict_data, speedups, CLI)
                dict_data = compute_sorts(dict_data)  # .sort_index()
                read_files.append(_base_name)

        elif file.endswith(".coz"):
            try:
                f = open(file, "r")
            except IOError as e:
                sys.stderr.write(f"{e}\n")
                continue

            lines = f.readlines()
            # Parse lines
            experiment = None

            for line in lines:
                if line != "\n":
                    isExperiment = False
                    data_type = ""
                    value = {}
                    sections = line.split("\t")
                    if len(sections) == 0:
                        continue
                    value["type"] = sections[0]
                    for section in sections:
                        splice = section.split("\n")[0].split("=")
                        if len(splice) > 1:
                            val = getValue(splice)
                            if isinstance(val, str) and "/" in val:
                                val = val[val.rfind("/") + 1 :]
                            value[splice[0]] = val
                        else:
                            data_type = splice[0]
                    if data_type == "experiment":
                        experiment = value
                    elif data_type == "throughput-point" or data_type == "progress-point":
                        experiment["speedup"] = 100 * experiment["speedup"]
                        data = addThroughput(data, experiment, value)
                    elif data_type == "latency-point":
                        data = addLatency(data, experiment, value)
                    elif data_type == "samples":
                        if value["location"] not in sample_data:
                            sample_data[value["location"]] = 0
                        sample_data[value["location"]] += int(value["count"])
                    elif data_type not in ["startup", "shutdown", "runtime"]:
                        print("Invalid profile")

            out = getSpeedupData(data.sort_index())
            read_files.append(_base_name)

    raise RuntimeError("foo")
    samples = pd.DataFrame({"samples": [sample_data]})
    return (
        pd.concat([out, dict_data, samples])
        if dict_data is not None
        else pd.concat([out, samples])
    )


def parseUploadedFile(file, experiments=".*", progress_points=".*"):
    data = pd.DataFrame()
    if "{" in file:
        raise RuntimeError("bar")
        dict_data = {}
        sample_data = {}
        _data = json.loads(file)
        dict_data = process_data(dict_data, _data, experiments, progress_points)
        sample_data = process_samples(sample_data, _data)
        data = compute_sorts(compute_speedups(dict_data))
        samples = pd.DataFrame({"samples": [sample_data]})
        data = pd.concat([data, samples])

    else:
        # coz
        for line in file.split("\n"):
            if len(line) > 0:
                isExperiment = False
                data_type = ""
                value = {}
                sections = line.split("\t")
                value["type"] = sections[0]
                for section in sections:
                    splice = section.split("\n")[0].split("=")
                    if len(splice) > 1:
                        value[splice[0]] = getValue(splice)
                    else:
                        data_type = splice[0]
                if data_type == "experiment":
                    experiment = value
                elif data_type == "throughput-point" or data_type == "progress-point":
                    experiment["speedup"] = 100 * experiment["speedup"]
                    data = addThroughput(data, experiment, value)
                elif data_type == "latency-point":
                    data = addLatency(data, experiment, value)
                elif data_type not in ["startup", "shutdown", "samples", "runtime"]:
                    print("Invalid profile")
                    # sys.exit(1)
        data = getSpeedupData(data.sort_index())
    return data


def getDataPoint(data):
    val = ""
    if math.isclose(data.delta, 0, rel_tol=1e-09, abs_tol=0.0):
        return np.nan
    if data["type"] == "throughput":
        val = data.duration / data.delta

        return val
    elif data["type"] == "latency":
        arrivalRate = data.arrivals / data.duration
        val = data.difference / arrivalRate
    else:
        print("invalid datapoint")
        val = np.inf
    if math.isinf(val):
        return np.nan
    else:
        return val


def getSpeedupData(data):
    cur_data = data.iloc[0]
    baseline_data_point = getDataPoint(data.iloc[0])
    speedup_df = pd.DataFrame()
    curr_selected = ""
    for index, row in data.iterrows():
        if curr_selected != index[0]:
            curr_selected = index[0]
            baseline_data_point = getDataPoint(row)

        maximize = True

        if row["type"] == "latency":
            maximize = False

        if not math.isnan(baseline_data_point):
            # for speedup in row:
            data_point = getDataPoint(row)
            if not math.isnan(data_point):
                progress_speedup = (
                    baseline_data_point - data_point
                ) / baseline_data_point
                # speedup = row["speedup"]
                speedup = row.name[2]

                name = row.name[0]

            if not maximize:
                # We are trying to *minimize* this progress point, so negate the speedup.
                progress_speedup = -progress_speedup

            if progress_speedup >= -1 and progress_speedup <= 2:
                speedup_df = pd.concat(
                    [
                        pd.DataFrame.from_dict(
                            data={
                                "point": [name],
                                "speedup": [speedup],
                                "progress_speedup": [100 * progress_speedup],
                            }
                        ),
                        speedup_df,
                    ]
                )
    speedup_df = speedup_df.sort_values(by=["speedup"])
    return speedup_df


def metadata_diff(json1, json2):
    res = jsondiff.diff(json1, json2)
    if res:
        print("Diff found")
    else:
        print("Same")
