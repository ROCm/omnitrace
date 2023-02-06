import sys
import json
import jsondiff
import pandas as pd
import numpy as np
from math import isnan, isinf, isclose
from collections import OrderedDict


class experiment_data(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.duration = []
        self.delta = []

    def __iadd__(self, _val):
        self.duration += [float(_val[0])]
        self.delta += [float(_val[1])]

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def mean(self):
        # print(self.duration)
        return sum(self.duration) / float(len(self.duration))

    def thorughput(self):
        if len(self.delta) < 1:
            return float("nan")
        return sum(self.duration) / sum(self.delta)


class line_speedup(object):
    def __init__(self, _name="", _prog="", _exp_data=None, _exp_base=None):
        self.name = _name
        self.prog = _prog
        self.data = _exp_data
        self.base = _exp_base

    def get(self):
        if self.data is None or self.base is None:
            return 0.0
        if len(self.data.delta) > 0:
            return (
                (self.base.thorughput() - self.data.thorughput()) / self.base.thorughput()
            ) * 100
        return ((self.base.mean() - self.data.mean()) / self.base.mean()) * 100

    def __str__(self):
        if self.data is None or self.base is None:
            return f"{self.name}"
        _line_speedup = self.get()
        return f"[{self.name}][{self.prog}][{self.data.speedup:3}] speedup: {_line_speedup:6.1f} %"

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


def find_or_insert(_data, _value):
    if _value not in _data:
        _data[_value] = experiment_data(_value)
    return _data[_value]


def longest_common_prefix(_inp):
    a = list(set(_inp))
    # if size is 0, return empty string
    # if size is 1, return first string
    if len(a) < 2:
        return ""
    # sort the array of strings
    a.sort()
    # if all the same, return empty string
    if a[0] == a[-1]:
        return ""
    # find the minimum length from first and last string
    end = min(len(a[0]), len(a[-1]))
    # find the common prefix between the first and last string
    i = 0
    while i < end and a[0][i] == a[-1][i]:
        i += 1
    pre = a[0][0:i]
    return pre


def process_data(data, _data):
    if not _data:
        return data
    for record in _data["omnitrace"]["causal"]["records"]:
        for exp in record["experiments"]:
            _speedup = exp["virtual_speedup"]
            _duration = exp["duration"]
            _file = exp["selection"]["info"]["file"]
            _line = exp["selection"]["info"]["line"]
            _func = exp["selection"]["info"]["dfunc"]
            _sym_addr = exp["selection"]["symbol_address"]
            _selected = ":".join([_file, f"{_line}"]) if _sym_addr == 0 else _func
            if _selected not in data:
                data[_selected] = {}
            for pts in exp["progress_points"]:
                _name = pts["name"]
                if _name not in data[_selected]:
                    data[_selected][_name] = {}
                itr = find_or_insert(data[_selected][_name], _speedup)
                if "delta" in pts:
                    _delt = pts["delta"]
                else:
                    _delt = pts["laps"]
                if _delt > 0:
                    # _delt=1
                    itr += [float(_duration), float(_delt)]
                else:
                    _diff = pts["arrival"] - pts["departure"] + 1
                    _rate = pts["arrival"] / float(_duration)
                    if _rate != 0:
                        itr += float(_diff) / float(_rate)
                # else:
                #    _delt = pts["laps"]
    return data


def compute_speedups(_data, CLI):
    data = {}
    out = pd.DataFrame()
    for selected, pitr in _data.items():
        if selected not in data:
            data[selected] = {}
        for progpt, ditr in pitr.items():
            data[selected][progpt] = OrderedDict(sorted(ditr.items()))
    from os.path import dirname

    ret = []
    _slcp = longest_common_prefix(
        [dirname(x.split(":")[0]) if ":" in x else x for x in _data.keys()]
    )
    for selected, pitr in _data.items():
        selected = selected.replace(_slcp, "", 1) if len(_slcp) > 0 else selected
        if len(selected) > 0 and selected[0] == "/":
            selected = selected[1:]
        _plcp = longest_common_prefix(
            [dirname(x.split(":")[0]) if ":" in x else x for x in pitr.keys()]
        )
        for progpt, ditr in pitr.items():
            progpt = progpt.replace(_plcp, "", 1) if len(_plcp) > 0 else progpt
            if len(progpt) > 0 and progpt[0] == "/":
                progpt = progpt[1:]
            if 0 not in ditr.keys() or len(ditr[0]) == 0:
                # print(f"missing baseline data for {progpt} in {selected}...")
                continue
            _baseline = ditr[0].mean()
            for speedup, itr in ditr.items():
                if speedup != itr.speedup:
                    raise ValueError(f"in {selected}: {speedup} != {itr.speedup}")
                _val = line_speedup(selected, progpt, itr, ditr[0])
                ret.append(_val)
    ret.sort()
    _last_name = None
    _last_prog = None
    for itr in ret:
        if CLI:
            if itr.name != _last_name or itr.prog != _last_prog:
                print("")
            # print(f"{itr}")
        if len(itr.data.duration) != 0:
            if itr.get() <= 200 and itr.get() >= -100:
                out = pd.concat(
                    [
                        out,
                        pd.DataFrame(
                            {
                                "idx": [(itr.prog, itr.name)],
                                "progress points": [itr.prog],
                                "point": [itr.name],
                                "Line Speedup": [itr.data.speedup],
                                "Program Speedup": [itr.get()],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        _last_name = itr.name
        _last_prog = itr.prog
    return out


def compute_sorts(_data):
    Max_speedup_order = _data.sort_values(
        by="Program Speedup", ascending=False
    ).point.unique()
    Min_speedup_order = _data.sort_values(
        by="Program Speedup", ascending=True
    ).point.unique()
    impactOrder = pd.DataFrame(_data.point.unique(), columns=["progress points"])
    point_counts = _data.point.value_counts()

    for index_imp, curr in impactOrder.iterrows():
        prev = pd.Series(dtype="float64")
        data_subset = _data[_data["point"] == curr["progress points"]]
        area = 0
        max_norm_area = 0
        for index_sub, data_point in data_subset.iterrows():
            if data_point["Line Speedup"] == 0:
                continue
            if prev.empty:
                prev = data_point
            else:
                avg_progress_speedup = (
                    prev["Program Speedup"] + data_point["Program Speedup"]
                ) / 2
                area = area + avg_progress_speedup * (
                    data_point["Line Speedup"] - prev["Line Speedup"]
                )
                norm_area = area / data_point["Line Speedup"]
                if norm_area > max_norm_area:
                    max_norm_area = norm_area
                prev = data_point
        impactOrder.at[index_imp, "area"] = max_norm_area
    impactOrder = impactOrder.sort_values(by="area")
    impactOrder = impactOrder["progress points"].unique()
    _data["Max Speedup"] = np.nan
    _data["Min Speedup"] = np.nan
    _data["impact"] = np.nan
    _data["point count"] = np.nan
    for index in _data.index:
        _data.at[index, "Max Speedup"] = np.where(
            Max_speedup_order == _data.at[index, "point"]
        )[0][0]
        _data.at[index, "impact"] = np.where(impactOrder == _data.at[index, "point"])[0][
            0
        ]
        _data.at[index, "Min Speedup"] = np.where(
            Min_speedup_order == _data.at[index, "point"]
        )[0][0]
        _data.at[index, "point count"] = point_counts[_data.at[index, "point"]]
    return _data


def isValidDataPoint(data):
    return isnan(data) == False and isinf(data) == False


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


def reformat_data(dict_data):
    df = pd.DataFrame(dict_data)
    out = pd.DataFrame()
    for key, value in df.iteritems():
        # print(key)
        _speedup = []
        for speedup, itr in value.items():
            out = pd.concat(
                [out, pd.DataFrame(data={"point": [key], "speedup": [speedup]})]
            )
        what = pd.DataFrame(value)
        # df[i] = j


def parseFiles(files, CLI):
    data = pd.DataFrame()
    out = pd.DataFrame()
    dict_data = {}

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
                dict_data = process_data(dict_data, _data)
                dict_data = compute_sorts(
                    compute_speedups(dict_data, CLI)
                )  # .sort_index()
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
                    elif data_type not in ["startup", "shutdown", "samples", "runtime"]:
                        print("Invalid profile")

            out = getSpeedupData(data.sort_index())
            read_files.append(_base_name)

    return pd.concat([out, dict_data]) if dict_data is not None else pd.concat([out])


def parseUploadedFile(file, CLI):
    data = pd.DataFrame()
    if "{" in file:
        dict_data = {}
        data_experiments = json.loads(file)
        dict_data = process_data(dict_data, data_experiments)
        data = compute_sorts(compute_speedups(dict_data, CLI))

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
    if isclose(data.delta, 0, rel_tol=1e-09, abs_tol=0.0):
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
    if isinf(val):
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

        if not isnan(baseline_data_point):
            # for speedup in row:
            data_point = getDataPoint(row)
            if not isnan(data_point):
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
                            },
                        ),
                        speedup_df,
                    ],
                )
    speedup_df = speedup_df.sort_values(by=["speedup"])
    return speedup_df


def metadata_diff(json1, json2):
    res = jsondiff.diff(json1, json2)
    if res:
        print("Diff found")
    else:
        print("Same")
