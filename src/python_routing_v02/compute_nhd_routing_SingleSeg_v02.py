#!/usr/bin/env python
# coding: utf-8
# example usage: python compute_nhd_routing_SingleSeg.py -v -t -w -n Mainstems_CONUS


# -*- coding: utf-8 -*-
"""NHD Network traversal

A demonstration version of this code is stored in this Colaboratory notebook:
    https://colab.research.google.com/drive/1ocgg1JiOGBUl3jfSUPCEVnW5WNaqLKCD

"""
## Parallel execution
import os
import sys
import time
import numpy as np
import argparse
import pathlib
import pandas as pd
from functools import partial
from joblib import delayed, Parallel
from itertools import chain, islice
from operator import itemgetter


def _handle_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debuglevel",
        help="Set the debuglevel",
        dest="debuglevel",
        choices=[0, -1, -2, -3],
        default=0,
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output (leave blank for quiet output)",
        dest="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--nts",
        "--number-of-qlateral-timesteps",
        help="Set the number of timesteps to execute. If used with ql_file or ql_folder, nts must be less than len(ql) x qN.",
        dest="nts",
        default=144,
        type=int,
    )
    parser.add_argument(
        "--sts",
        "--assume-short-ts",
        help="Use the previous timestep value for upstream flow",
        dest="assume_short_ts",
        action="store_true",
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        help="Use the parallel computation engine (omit flag for serial computation)",
        dest="parallel_compute",
        const="type3",
    )
    parser.add_argument(
        "--cpu-pool",
        help="Assign the number of cores to multiprocess across.",
        dest="cpu_pool",
        type=int,
        default="-1",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write output files (leave blank for no writing)",
        dest="write_output",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--showtiming",
        help="Set the showtiming (leave blank for no timing information)",
        dest="showtiming",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--break-at-waterbodies",
        help="Use the waterbodies in the route-link dataset to divide the computation (leave blank for no splitting)",
        dest="break_network_at_waterbodies",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--supernetwork",
        help="Choose from among the pre-programmed supernetworks (Pocono_TEST1, Pocono_TEST2, Pocono_TEST_tiny, LowerColorado_Conchos_FULL_RES, Brazos_LowerColorado_ge5, Brazos_LowerColorado_FULL_RES, Brazos_LowerColorado_Named_Streams, CONUS_ge5, Mainstems_CONUS, CONUS_Named_Streams, CONUS_FULL_RES_v20",
        choices=[
            "Pocono_TEST1",
            "Pocono_TEST2", 
            "Pocono_TEST_tiny",
            "LowerColorado_Conchos_FULL_RES",
            "Brazos_LowerColorado_ge5",
            "Brazos_LowerColorado_FULL_RES",
            "Brazos_LowerColorado_Named_Streams",
            "CONUS_ge5",
            "Mainstems_CONUS",
            "CONUS_Named_Streams",
            "CONUS_FULL_RES_v20",
        ],
        # TODO: accept multiple or a Path (argparse Action perhaps)
        # action='append',
        # nargs=1,
        dest="supernetwork",
        default="Pocono_TEST1",
    )
    parser.add_argument("--ql", help="QLat input data", dest="ql", default=None)

    return parser.parse_args()


ENV_IS_CL = False
if ENV_IS_CL:
    root = pathlib.Path("/", "content", "t-route")
elif not ENV_IS_CL:
    root = pathlib.Path("../..").resolve()
    sys.path.append(r"../python_framework_v02")
    sys.path.append(r"./fast_reach")

    # TODO: automate compile for the package scripts
    # sys.path.append(r"../fortran_routing/mc_pylink_v00/MC_singleSeg_singleTS")

## network and reach utilities
import nhd_network_utilities_v02 as nnu
import mc_reach
import mc_reach_py
import nhd_network
import nhd_io


def writetoFile(file, writeString):
    file.write(writeString)
    file.write("\n")


def constant_qlats(data, nsteps, qlat):
    q = np.full((len(data.index), nsteps), qlat, dtype="float32")
    ql = pd.DataFrame(q, index=data.index, columns=range(nsteps))
    return ql


def main():

    args = _handle_args()

    nts = args.nts
    debuglevel = -1 * args.debuglevel
    verbose = args.verbose
    showtiming = args.showtiming
    supernetwork = args.supernetwork
    break_network_at_waterbodies = args.break_network_at_waterbodies
    write_output = args.write_output
    assume_short_ts = args.assume_short_ts

    test_folder = pathlib.Path(root, "test")
    geo_input_folder = test_folder.joinpath("input", "geo")

    # TODO: Make these commandline args
    """##NHD Subset (Brazos/Lower Colorado)"""
    # supernetwork = 'Brazos_LowerColorado_Named_Streams'
    # supernetwork = 'Brazos_LowerColorado_ge5'
    # supernetwork = 'Pocono_TEST1'
    """##NHD CONUS order 5 and greater"""
    # supernetwork = 'CONUS_ge5'
    """These are large -- be careful"""
    # supernetwork = 'Mainstems_CONUS'
    # supernetwork = 'CONUS_FULL_RES_v20'
    # supernetwork = 'CONUS_Named_Streams' #create a subset of the full resolution by reading the GNIS field
    # supernetwork = 'CONUS_Named_combined' #process the Named streams through the Full-Res paths to join the many hanging reaches

    if verbose:
        print("creating supernetwork connections set")
    if showtiming:
        start_time = time.time()

    # STEP 1
    network_data = nnu.set_supernetwork_data(
        supernetwork=args.supernetwork,
        geo_input_folder=geo_input_folder,
        verbose=False,
        debuglevel=debuglevel,
    )

    cols = network_data["columns"]
    param_df = nhd_io.read(network_data["geo_file_path"])
    param_df = param_df[list(cols.values())]
    param_df = param_df.set_index(cols["key"])

    if "mask_file_path" in network_data:
        data_mask = nhd_io.read_mask(
            network_data["mask_file_path"],
            layer_string=network_data["mask_layer_string"],
        )
        param_df = param_df.filter(data_mask.iloc[:, network_data["mask_key"]], axis=0)

    param_df = param_df.sort_index()
    param_df = nhd_io.replace_downstreams(param_df, cols["downstream"], 0)

    if args.ql:
        qlats = nhd_io.read_qlat(args.ql)
    else:
        qlats = constant_qlats(param_df, nts, 10.0)

    connections = nhd_network.extract_connections(param_df, cols["downstream"])
    wbodies = nhd_network.extract_waterbodies(
        param_df, cols["waterbody"], network_data["waterbody_null_code"]
    )

    if verbose:
        print("supernetwork connections set complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    # STEP 2
    if showtiming:
        start_time = time.time()
    if verbose:
        print("organizing connections into reaches ...")

    rconn = nhd_network.reverse_network(connections)
    subnets = nhd_network.reachable_network(rconn)
    reaches_bytw = {}
    ordered_reaches = {}
    tuple_reaches = {}
    for tw, net in subnets.items():
        path_func = partial(nhd_network.split_at_junction, net)
        reaches_bytw[tw] = nhd_network.dfs_decomposition(net, path_func)
        ordered_reaches[tw] = nhd_network.dfs_decomposition_depth2(net, path_func)
        tuple_reaches[tw] = nhd_network.dfs_decomposition_depth_tuple(net, path_func)
    
    #TODO: instead of operating on the overall_ordered_reaches_list below, see if 
    # we can directly use this list of tuples directly, which contains all the ordering
    # information otherwise.
    overall_tuple_reaches = []
    for _, tuple_list in tuple_reaches.items():
        overall_tuple_reaches.extend(tuple_list)

    overall_ordered_reaches_dict = nhd_network.tuple_with_orders_into_dict(overall_tuple_reaches)
    max_order = max(overall_ordered_reaches_dict.keys())

    overall_ordered_reaches_list = []
    ordered_reach_count = []
    ordered_reach_cache_count = []
    for o in range(max_order,-1,-1):
        overall_ordered_reaches_list.extend(overall_ordered_reaches_dict[o])
        ordered_reach_count.append(len(overall_ordered_reaches_dict[o]))
        ordered_reach_cache_count.append(sum(len(r) for r in overall_ordered_reaches_dict[o]))

    rconn_ordered = {}
    rconn_ordered_byreach = {}
    for o in range(max(overall_ordered_reaches_dict.keys()),0,-1):
        rconn_ordered[o] = {}
        for reach in overall_ordered_reaches_dict[o]:
            for segment in reach:
                rconn_ordered[o][segment] = rconn[segment]
                rconn_ordered_byreach[segment] = rconn[segment]

    if verbose:
        print("reach organization complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    if showtiming:
        start_time = time.time()

    param_df["dt"] = 300.0
    param_df = param_df.rename(columns=nnu.reverse_dict(cols))
    param_df = param_df.astype("float32")

    # datasub = data[['dt', 'bw', 'tw', 'twcc', 'dx', 'n', 'ncc', 'cs', 's0']]

    parallel_compute = args.parallel_compute
    if parallel_compute=="type2":
        print("Executing in Parallel type 2 mode (thread pool shared across reaches)")
        print("Communication between reaches handled by python framework")
        with Parallel(n_jobs=args.cpu_pool, backend="threading") as parallel:
            jobs = []
            for o in range(max_order,0,-1):
                reach_list = overall_ordered_reaches_dict[o] 
                r = list(chain.from_iterable(reach_list))
                param_df_sub = param_df.loc[
                    r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
                ].sort_index()
                qlat_sub = qlats.loc[r].sort_index()
                jobs.append(
                    delayed(mc_reach_py.compute_network)(
                        nts,
                        reach_list,
                        rconn_ordered[o],
                        param_df_sub.index.values,
                        param_df_sub.columns.values,
                        param_df_sub.values,
                        qlat_sub.values,
                        np.array([len(reach_list)], dtype="int32"),
                        np.array([sum(len(r) for r in reach_list)], dtype="int32"),
                        assume_short_ts,
                    )
                )
            results = parallel(jobs)

    elif parallel_compute=="type1":
        print("Executing in Parallel type 1 mode (1 thread per independent basin)")
        with Parallel(n_jobs=args.cpu_pool, backend="threading") as parallel:
            jobs = []
            for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
                r = list(chain.from_iterable(reach_list))
                param_df_sub = param_df.loc[
                    r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
                ].sort_index()
                qlat_sub = qlats.loc[r].sort_index()
                jobs.append(
                    delayed(mc_reach_py.compute_network)(
                        nts,
                        reach_list,
                        subnets[tw],
                        param_df_sub.index.values,
                        param_df_sub.columns.values,
                        param_df_sub.values,
                        qlat_sub.values,
                        np.array([len(reach_list)], dtype="int32"),
                        np.array([sum(len(r) for r in reach_list)], dtype="int32"),
                        assume_short_ts,
                    )
                )
            results = parallel(jobs)

    elif parallel_compute=="type3":
        print("Executing in Parallel type 3 mode:")
        print("(type3 = thread pool shared across reaches and ")
        print("communication between reaches handled by cython framework)")

        r = list(chain.from_iterable(overall_ordered_reaches_list))
        param_df_sub = param_df.loc[
            r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
        ].sort_index()
        qlat_sub = qlats.loc[r].sort_index()
        results = mc_reach_py.compute_network(
            nts,
            overall_ordered_reaches_list,
            rconn_ordered_byreach,
            param_df_sub.index.values,
            param_df_sub.columns.values,
            param_df_sub.values,
            qlat_sub.values,
            np.array(ordered_reach_count, dtype="int32"),
            np.array(ordered_reach_cache_count, dtype="int32"),
            assume_short_ts,
        )

    elif parallel_compute == "serial_reach":
        print("Executing in Reach-centric Serial mode")
        results = []
        for o in range(max_order,0,-1):
            reach_list = overall_ordered_reaches_dict[o] 
            r = list(chain.from_iterable(reach_list))
            param_df_sub = param_df.loc[
                r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
            ].sort_index()
            qlat_sub = qlats.loc[r].sort_index()
            results.append(
                mc_reach_py.compute_network(
                    nts,
                    reach_list,
                    rconn_ordered[o],
                    param_df_sub.index.values,
                    param_df_sub.columns.values,
                    param_df_sub.values,
                    qlat_sub.values,
                    np.array([len(reach_list)], dtype="int32"),
                    np.array([sum(len(r) for r in reach_list)], dtype="int32"),
                    assume_short_ts,
                )
            )

    else:
        print("Executing in Network-centric Serial mode")
        results = []
        for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
            r = list(chain.from_iterable(reach_list))
            param_df_sub = param_df.loc[
                r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
            ].sort_index()
            qlat_sub = qlats.loc[r].sort_index()
            results.append(
                mc_reach_py.compute_network(
                    nts,
                    reach_list,
                    subnets[tw],
                    param_df_sub.index.values,
                    param_df_sub.columns.values,
                    param_df_sub.values,
                    qlat_sub.values,
                    np.array([len(reach_list)], dtype="int32"),
                    np.array([sum(len(r) for r in reach_list)], dtype="int32"),
                    assume_short_ts,
                )
            )

    fdv_columns = pd.MultiIndex.from_product(
        [range(nts), ["q", "v", "d"]]
    ).to_flat_index()
    if parallel_compute == "type3":
        #TODO: Why does this not just work with the else-condition code?
        #Testing with the 'type3' execution yields baffling results... but this works for now.
        flowveldepth = pd.DataFrame(results[1], index=results[0], columns=fdv_columns)
    else:
        flowveldepth = pd.concat(
            [pd.DataFrame(d, index=i, columns=fdv_columns) for i, d in results], copy=False
        )
    flowveldepth = flowveldepth.sort_index()
    flowveldepth.to_csv(f"{args.supernetwork}.csv")
    #print(flowveldepth.loc[[4186169]])
    print(flowveldepth)

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))


if __name__ == "__main__":
    main()
