"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
import sqlite3
import argparse
import traceback


class DatabaseManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.connection = None
        self.cursor = None

    def open_connection(self):
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def execute_sql(self, sql):
        try:
            self.cursor.execute(sql)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Execute sql '{sql}' error: {e}")
            traceback.print_exc()


def are_tables_exist(db_manager, table_names):
    try:
        # Query for the existence of sqlite database tables with specific names
        results = {}
        for table_name in table_names:
            db_manager.execute_sql(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            )
            result = db_manager.cursor.fetchone()
            results[table_name] = result is not None
        return results

    except sqlite3.Error as e:
        print(f"are_tables_exist() SQLite error: {e}")
        return {}


def print_db_info(db_manager):
    # execute sql
    db_manager.execute_sql("SELECT name, sql FROM sqlite_master WHERE type='table';")
    # get results
    tables = db_manager.cursor.fetchall()
    # print infomation
    for table in tables:
        print(f"Table Name: {table[0]}\nCreate Table SQL: {table[1]}\n")


def get_start_time(db_manager):
    """
    get session start time(timestamp) from table TARGET_INFO_SESSION_START_TIME
    """
    sql = "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1;"
    db_manager.execute_sql(sql)
    result = db_manager.cursor.fetchone()
    timestamp = result[0]
    return timestamp


def get_process_id(db_manager):
    """
    get process id from table TARGET_INFO_CUDA_NULL_STREAM
    """
    sql = "SELECT processId FROM TARGET_INFO_CUDA_NULL_STREAM LIMIT 1;"
    db_manager.execute_sql(sql)
    result = db_manager.cursor.fetchone()
    process_id = result[0]
    return process_id


def get_device_property(db_manager):
    """
    get device properties from TARGET_INFO_GPU
    """
    sql = (
        "SELECT name,totalMemory,computeMajor,computeMinor,"
        "maxThreadsPerBlock,maxBlocksPerSm,maxRegistersPerBlock,"
        "maxRegistersPerSm,threadsPerWarp,maxShmemPerBlock,"
        "maxRegistersPerSm,smCount,maxShmemPerBlockOptin "
        "FROM TARGET_INFO_GPU WHERE id is 0;"
    )
    db_manager.execute_sql(sql)
    (
        name,
        totalGlobalMem,
        computeMajor,
        computeMinor,
        maxThreadsPerBlock,
        maxBlocksPerSm,
        regsPerBlock,
        regsPerMultiprocessor,
        warpSize,
        sharedMemPerBlock,
        sharedMemPerMultiprocessor,
        numSms,
        sharedMemPerBlockOptin,
    ) = db_manager.cursor.fetchone()
    maxThreadsPerMultiprocessor = maxThreadsPerBlock * maxBlocksPerSm

    property = {
        "id": 0,
        "name": name,
        "totalGlobalMem": totalGlobalMem,
        "computeMajor": computeMajor,
        "computeMinor": computeMinor,
        "maxThreadsPerBlock": maxThreadsPerBlock,
        "maxThreadsPerMultiprocessor": maxThreadsPerMultiprocessor,
        "regsPerBlock": regsPerBlock,
        "regsPerMultiprocessor": regsPerMultiprocessor,
        "warpSize": warpSize,
        "sharedMemPerBlock": sharedMemPerBlock,
        "sharedMemPerMultiprocessor": sharedMemPerMultiprocessor,
        "numSms": numSms,
        "sharedMemPerBlockOptin": sharedMemPerBlockOptin,
    }
    return property


def sqlite_to_google_trace_event(args, tables):
    try:
        database_path = args.input
        print("Opening sqlite database :", database_path)
        db_manager = DatabaseManager(database_path)
        db_manager.open_connection()

        # print basic database information
        if args.info:
            print_db_info(db_manager)

        print("Checking if the following table exists:")
        results = are_tables_exist(db_manager, tables)
        for table_name, exists in results.items():
            if not exists:
                print(f"'{table_name}' not exists.")
                raise ValueError(
                    f"Table '{table_name}' does not exist in the database."
                )
            else:
                print(f"'{table_name}' exists.")

        # get some necessary information
        session_start_time = get_start_time(db_manager)  # session start time
        process_id = get_process_id(db_manager)  # process id
        device_property = get_device_property(db_manager)  # properties of cuda device

        deviceProperties = [device_property]
        db_manager.execute_sql(
            "SELECT name,busLocation FROM TARGET_INFO_GPU WHERE id is 0;"
        )
        name, bus_location = db_manager.cursor.fetchone()
        db_manager.execute_sql(
            "SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS LIMIT 1;"
        )
        trace_duration, trace_start_time, trace_stop_time = db_manager.cursor.fetchone()

        raw_start_time = session_start_time + trace_start_time
        start_time = round(raw_start_time / 1000)  # μs to ms
        end_time = round((session_start_time + trace_stop_time) / 1000)
        duration = round(trace_duration / 1000)  # μs to ms
        traceEvents_data = []
        # construct process meta infomations
        traceEvents_meta = [
            {
                "name": "process_name",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": 0,
                "args": {"name": "python3"},
            },
            {
                "name": "process_labels",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": 0,
                "args": {"labels": "CPU"},
            },
            {
                "name": "process_sort_index",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": 0,
                "args": {"sort_index": process_id},
            },
            {
                "name": "process_name",
                "ph": "M",
                "ts": start_time,
                "pid": 0,
                "tid": 0,
                "args": {"name": "python3"},
            },
            {
                "name": "process_labels",
                "ph": "M",
                "ts": start_time,
                "pid": 0,
                "tid": 0,
                "args": {"labels": f"GPU 0(CUDA HW {bus_location} - {name})"},
            },
            {
                "name": "process_sort_index",
                "ph": "M",
                "ts": start_time,
                "pid": 0,
                "tid": 0,
                "args": {"sort_index": process_id},
            },
            {
                "ph": "X",
                "cat": "Trace",
                "ts": start_time,
                "dur": duration,
                "pid": "Spans",
                "tid": "OneFlow Insight",
                "name": "OneFlow Insight (0)",
                "args": {"Op count": 0},
            },
            {
                "name": "process_sort_index",
                "ph": "M",
                "ts": start_time,
                "pid": "Spans",
                "tid": 0,
                "args": {"sort_index": "Spans"},
            },
            {
                "name": "Iteration Start: OneFlow Insight",
                "ph": "i",
                "s": "g",
                "pid": "Traces",
                "tid": "Trace OneFlow Insight",
                "ts": start_time,
            },
            {
                "name": "Record Window End",
                "ph": "i",
                "s": "g",
                "pid": "",
                "tid": "",
                "ts": end_time,
            },
        ]

        # construct vm threads meta infomations
        db_manager.execute_sql("SELECT text,globalTid FROM NVTX_EVENTS;")
        globalTids = []
        for row in db_manager.cursor.fetchall():
            text, globalTid = row
            globalTids.append(globalTid)
            osrt_name = {
                "name": "thread_name",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": f"[OSRT API]{globalTid}",
                "args": {"name": f"[OSRT API]{text}"},
            }
            osrt_sort_index = {
                "name": "thread_sort_index",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": f"[OSRT API]{globalTid}",
                "args": {"sort_index": globalTid - 1},
            }
            cu_api_name = {
                "name": "thread_name",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": globalTid,
                "args": {"name": f"[CUDA API]{text}"},
            }
            cu_api_name_index = {
                "name": "thread_sort_index",
                "ph": "M",
                "ts": start_time,
                "pid": process_id,
                "tid": globalTid,
                "args": {"sort_index": globalTid},
            }
            traceEvents_meta.append(osrt_name)
            traceEvents_meta.append(osrt_sort_index)
            traceEvents_meta.append(cu_api_name)
            traceEvents_meta.append(cu_api_name_index)

        # construct cuda stream meta infomations
        db_manager.execute_sql(
            "SELECT streamId,processId FROM TARGET_INFO_CUDA_STREAM;"
        )
        temp_time = start_time
        for row in db_manager.cursor.fetchall():
            temp_time += 187000
            streamId, processId = row
            thread_name = {
                "name": "thread_name",
                "ph": "M",
                "ts": start_time,
                "pid": 0,
                "tid": streamId,
                "args": {"name": f"cuda stream {streamId}", "stream": streamId,},
            }
            thread_sort_index = {
                "name": "thread_sort_index",
                "ph": "M",
                "ts": start_time,
                "pid": 0,
                "tid": streamId,
                "args": {"sort_index": streamId},
            }
            traceEvents_meta.append(thread_name)
            traceEvents_meta.append(thread_sort_index)

        # insert os runtime events
        global_tids = ", ".join(map(str, globalTids))
        db_manager.execute_sql(
            f"SELECT start,end,globalTid,nameId  FROM OSRT_API WHERE globalTid IN ({global_tids});"
        )
        for row in db_manager.cursor.fetchall():
            start, end, globalTid, nameId = row
            db_manager.execute_sql(f"SELECT value FROM StringIds WHERE id = {nameId};")
            name = db_manager.cursor.fetchone()[0]
            ts = (raw_start_time + start) / 1000
            dur = (end - start) / 1000
            row_data = {
                "ph": "X",
                "cat": "OS RUNTIME API",
                "name": name,
                "pid": process_id,
                "tid": f"[OSRT API]{globalTid}",
                "ts": ts,
                "dur": dur,
                "args": {"global tid": f"{globalTid}(serialized)",},
            }
            traceEvents_data.append(row_data)

        # insert cuda runtime api events
        db_manager.execute_sql(
            "SELECT start,end,globalTid,correlationId,nameId  FROM CUPTI_ACTIVITY_KIND_RUNTIME;"
        )
        for row in db_manager.cursor.fetchall():
            start, end, globalTid, correlationId, nameId = row
            db_manager.execute_sql(f"SELECT value FROM StringIds WHERE id is {nameId};")
            name = db_manager.cursor.fetchone()[0]
            short_name = name.split("_", 1)[0]
            ts = (raw_start_time + start) / 1000
            dur = (end - start) / 1000
            row_data = {
                "ph": "X",
                "cat": "CUDA API",
                "name": short_name,
                "pid": process_id,
                "tid": globalTid,
                "ts": ts,
                "dur": dur,
                "args": {
                    "name": f"Call to {name}",
                    "begins": f"{start/(10**9)}s",
                    "ends": f"{end/(10**9)}s(+{dur}ms)",
                    "global tid": f"{globalTid}(serialized)",
                    "correlation id": correlationId,
                },
            }
            traceEvents_data.append(row_data)

        # insert cuda kernel events
        db_manager.execute_sql(
            (
                "SELECT start,end,deviceId,contextId,streamId,"
                "correlationId,globalPid,demangledName,shortName,"
                "gridX,gridY,gridZ,blockX,blockY,blockZ,"
                "staticSharedMemory,dynamicSharedMemory,localMemoryTotal "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL;"
            )
        )
        for row in db_manager.cursor.fetchall():
            (
                start,
                end,
                deviceId,
                contextId,
                streamId,
                correlationId,
                globalPid,
                demangledName,
                shortName,
                gridX,
                gridY,
                gridZ,
                blockX,
                blockY,
                blockZ,
                staticSharedMemory,
                dynamicSharedMemory,
                localMemoryTotal,
            ) = row
            db_manager.execute_sql(
                f"SELECT value FROM StringIds WHERE id is {shortName}"
            )
            short_name = db_manager.cursor.fetchone()[0]
            db_manager.execute_sql(
                f"SELECT value FROM StringIds WHERE id is {demangledName}"
            )
            name = db_manager.cursor.fetchone()[0]
            ts = (raw_start_time + start) / 1000
            dur = (end - start) / 1000
            row_data = {
                "ph": "X",
                "cat": "CUDA Kernel",
                "name": short_name,
                "pid": 0,
                "tid": streamId,
                "ts": ts,
                "dur": dur,
                "args": {
                    "name": name,
                    "begins": f"{start/(10**9)}s",
                    "ends": f"{end/(10**9)}s(+{dur}ms)",
                    "grid": f"<<<{gridX},{gridY},{gridZ}>>>",
                    "block": f"<<<{blockX},{blockY},{blockZ}>>>",
                    "static shared memory": f"{staticSharedMemory}bytes",
                    "dynamic shared memory": f"{dynamicSharedMemory}bytes",
                    "local memory total": f"{localMemoryTotal}bytes",
                    "global pid": f"{globalPid}(serialized)",
                    "device id": deviceId,
                    "context id": contextId,
                    "stream id": streamId,
                    "correlation id": correlationId,
                },
            }

            traceEvents_data.append(row_data)

        # construct trace event dict
        traceEvents = traceEvents_data + traceEvents_meta
        data = {"deviceProperties": deviceProperties, "traceEvents": traceEvents}

        # the path to the JSON file to be written
        json_fpath = args.output

        # write dict content into a JSON file using json.dump
        with open(json_fpath, "w") as json_file:
            json.dump(data, json_file, indent=2)
        print(f"Successfully converted content to file: {json_fpath}")

    except BaseException as e:
        print(f"An exception occurred: {type(e).__name__}: {e}")
        traceback.print_exc()

    finally:
        # close db connection
        db_manager.close_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")

    parser.add_argument("--input", help="Input nvidia nsight system .sqlite file path")
    parser.add_argument(
        "--output",
        "-o",
        help="Output json file path(google trace format)",
        default="sqlite_to_google_trace_event.json",
    )
    parser.add_argument(
        "--info",
        "-v",
        action="store_true",
        help="Enable print infomation of sqlite database",
        default=False,
    )

    args = parser.parse_args()
    # check if necessary tables exist
    tables_to_check = [
        "TARGET_INFO_GPU",
        "TARGET_INFO_SESSION_START_TIME",
        "TARGET_INFO_CUDA_NULL_STREAM",
        "ANALYSIS_DETAILS",
        "NVTX_EVENTS",
        "TARGET_INFO_CUDA_STREAM",
        "OSRT_API",
        "StringIds",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_KERNEL",
    ]

    # Usage:
    # python3 sqlite_to_google_trace_event.py --input 'your_file.sqlite'
    sqlite_to_google_trace_event(args, tables_to_check)
