#!/usr/bin/env python3

import os
import sys
import json
import argparse
from collections import OrderedDict


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
        return sum(self.duration) / float(len(self.duration))


class line_speedup(object):
    def __init__(self, _name="", _prog="", _exp_data=None, _exp_base=None):
        self.name = _name
        self.prog = _prog
        self.data = _exp_data
        self.base = _exp_base

    def get(self):
        if self.data is None or self.base is None:
            return 0.0
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
                    if _delt > 0:
                        itr += float(_duration) / float(_delt)
                    else:
                        _diff = pts["arrival"] - pts["departure"] + 1
                        _rate = pts["arrival"] / float(_duration)
                        itr += float(_diff) / float(_rate)
                else:
                    _delt = pts["laps"]

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
            if 0 not in ditr.keys():
                print(f"missing baseline data for {progpt} in {selected}...")
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
        if itr.name != _last_name or itr.prog != _last_prog:
            print("")
        print(f"{itr}")
        _last_name = itr.name
        _last_prog = itr.prog

    return ret


if __name__ == "__main__":
    data = {}
    for inp in sys.argv[1:]:
        with open(inp, "r") as f:
            inp_data = json.load(f)
        data = process_data(data, inp_data)

    compute_speedups(data)
