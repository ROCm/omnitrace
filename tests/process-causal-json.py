#!/usr/bin/env python3

import os
import sys
import json
import argparse
from collections import OrderedDict


def mean(_data):
    return sum(_data) / float(len(_data)) if len(_data) > 0 else 0.0


def stddev(_data):
    if len(_data) == 0:
        return 0.0
    _mean = mean(_data)
    _variance = sum([((x - _mean) ** 2) for x in _data]) / float(len(_data))
    return _variance**0.5


class experiment_data(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.duration = []

    def __iadd__(self, _val):
        self.duration += [float(_val)]

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def mean(self):
        return mean(self.duration)

    def stddev(self):
        return stddev(self.duration)


class line_speedup(object):
    def __init__(self, _name="", _prog="", _exp_data=None, _exp_base=None):
        self.name = _name
        self.prog = _prog
        self.data = _exp_data
        self.base = _exp_base

    def get_speedup(self):
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
        for ditr in self.data.duration:
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
        _line_stddev = 1.0 * self.compute_speedup_stddev()  # 3 stddev == 99.87%
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
        """
        speedup_c = [x.compute_speedup() for x in self.data]
        speedup_v = [x.get_speedup() for x in self.data]
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
        return "\n".join(_impact + [f"{x}" for x in self.data])

    def __lt__(self, rhs):
        self.data.sort()
        return self.get_impact()[0] < rhs.get_impact()[0]


def find_or_insert(_data, _value):
    if _value not in _data:
        _data[_value] = experiment_data(_value)
    return _data[_value]


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
                if "delta" in pts:
                    _delt = pts["delta"]
                    if _delt > 0:
                        itr = find_or_insert(data[_selected][_name], _speedup)
                        itr += float(_duration) / float(_delt)
                    else:
                        _diff = pts["arrival"] - pts["departure"] + 1
                        _rate = pts["arrival"] / float(_duration)
                        if _rate > 0:
                            itr = find_or_insert(data[_selected][_name], _speedup)
                            itr += float(_diff) / float(_rate)
                else:
                    _delt = pts["laps"]
                    if _delt > 0:
                        itr = find_or_insert(data[_selected][_name], _speedup)
                        itr += float(_duration) / float(_delt)

    return data


def compute_speedups(_data):
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
        _data.append(experiment_progress(itr))

    _data.sort()
    for itr in _data:
        if len(itr) < 2:
            continue
        print("")
        print(f"{itr}")

    return _data


if __name__ == "__main__":
    data = {}
    for inp in sys.argv[1:]:
        with open(inp, "r") as f:
            inp_data = json.load(f)
        data = process_data(data, inp_data)

    compute_speedups(data)
