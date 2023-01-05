from math import isnan, isinf, isclose
import pandas as pd
import numpy as np
import sys
import json 
import jsondiff
def isValidDataPoint(data):
    return isnan(data) == False and isinf(data) == False

def getValue(splice):
    if "type" in splice:
        #Something might happen here later
        print("type found in coz file")
        sys.exit(1)
    if "difference" in splice or "speedup" in splice:
        return float(splice[1])
    else:
        return (splice[1])

def addThroughput(df, experiment, value):
    #maybe id is issue
    ix = (experiment["selected"], value["name"], experiment["speedup"])
    df_points = list(df.index)
    if ix not in df_points:
        new_row= pd.DataFrame(
            data={
                #"selected" : [experiment["selected"]],
                #"speedup" : [float(experiment["speedup"])],
                "delta" : [float(value["delta"])],
                "duration" : [float(experiment["duration"])],
                "type" : ["throughput"]
            },
            index = [ix]
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
                "selected" : [experiment["selected"]],
                "point" : [value["name"]],
                "speedup" : [float(experiment["speedup"])],
                "arrivals" : [float(value["arrivals"])],
                "departures" : [float(value["departures"])],
                "duration" : [0],
                "type" : ["latency"]
            }
        )
    
    if value.duration == 0:
        df["difference"] = value.difference
    else:
        duration = df["duration"][ix] + float(experiment["duration"])
        df["difference"][ix] = df["difference"][ix] * df["duration"][ix] / duration
        df["difference"][ix] = df["difference"] + (float(value["difference"]) * float(experiment["duration"])) / duration
    
    df["duration"] = df["duration"] + float(experiment["duration"])

    return df 

def parseFile(file):
    f = open(file, "r")
    lines = f.readlines()
    data=pd.DataFrame()
    if '{' in lines:
        data=pd.read_json(file)
    else:
        #Parse lines
        experiment=None
        
        for line in lines:
            if line != '\n':
                isExperiment = False
                data_type=""
                value={}
                sections = line.split("\t")
                value["type"] = sections[0]
                for section in sections:
                    splice=section.split("\n")[0].split("=")
                    if len(splice) > 1:
                        value[splice[0]] = getValue(splice)
                    else:
                        data_type=splice[0]
                if data_type == "experiment":
                    experiment = value
                elif data_type == 'throughput-point' or data_type == 'progress-point':
                    data = addThroughput(data, experiment, value)
                elif data_type == 'latency-point':
                    data = addLatency(data, experiment, value)
                elif (data_type not in ["startup", "shutdown", "samples", "runtime"]):
                    print("Invalid profile")
                    #sys.exit(1)
    return data

def parseUploadedFile(file):
    data=pd.DataFrame()
    for line in file.split("\n"):
        if len(line) >0:
            isExperiment = False
            data_type=""
            value={}
            sections = line.split("\t")
            value["type"] = sections[0]
            for section in sections:
                splice=section.split("\n")[0].split("=")
                if len(splice) > 1:
                    value[splice[0]] = getValue(splice)
                else:
                    data_type=splice[0]
            if data_type == "experiment":
                experiment = value
            elif data_type == 'throughput-point' or data_type == 'progress-point':
                data = addThroughput(data, experiment, value)
            elif data_type == 'latency-point':
                data = addLatency(data, experiment, value)
            elif (data_type not in ["startup", "shutdown", "samples", "runtime"]):
                print("Invalid profile")
                #sys.exit(1)
    return data.sort_index()
def getDataPoint(data):
    val = ""
    if isclose(data.delta, 0, rel_tol=1e-09, abs_tol=0.0):
        return np.nan
    if data["type"] == "throughput":
        val =  data.duration / data.delta
        
        return val
    elif data["type"] == "latency":
        arrivalRate = data.arrivals / data.duration
        val =  data.difference / arrivalRate
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
    speedup_df=pd.DataFrame()
    curr_selected =""
    for index, row in data.iterrows():
        if curr_selected != index[0]:
            curr_selected = index[0]
            baseline_data_point = getDataPoint(row)

        maximize = True

        if row["type"] == "latency":
            maximize = False

        if not isnan(baseline_data_point):
            #for speedup in row:
            data_point = getDataPoint(row)
            if not isnan(data_point):
                    
                progress_speedup = (baseline_data_point - data_point) / baseline_data_point
                #speedup = row["speedup"]
                speedup=row.name[2]
                
                name = row.name[0]

            if (not maximize):
                    #We are trying to *minimize* this progress point, so negate the speedup.
                progress_speedup = -progress_speedup
                    
            if (progress_speedup >= -1 and progress_speedup <= 2):
                speedup_df=pd.concat(
                    [pd.DataFrame.from_dict(
                    data={"point":[name], "speedup":[speedup], "progress_speedup":[progress_speedup]},
                    ),speedup_df],
                )
    speedup_df = speedup_df.sort_values(by=["speedup"])
    return speedup_df
def metadata_diff(json1, json2):
    res = jsondiff.diff(json1, json2)
    if res:
        print("Diff found")
    else:
        print("Same")