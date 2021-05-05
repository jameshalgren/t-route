from collections import defaultdict
from itertools import chain
from functools import partial
from joblib import delayed, Parallel
import time
import pandas as pd
import numpy as np

import troute.nhd_network as nhd_network
import fast_reach
from .fast_reach import diffusive

_compute_func_map = defaultdict(
        fast_reach.compute_network,
        {
            "diffusive": diffusive.compute_diffusive_tst,
            "V02-caching": fast_reach.compute_network,
            "V02-structured": fast_reach.compute_network_structured,
            "V02-structured-obj": fast_reach.compute_network_structured_obj
        }
    )

def compute_nhd_routing_v02(
    connections,
    rconn,
    wbodies,
    reaches_bytw,
    compute_func_name,
    parallel_compute_method,
    subnetwork_target_size,
    cpu_pool,
    nts,
    qts_subdivisions,
    independent_networks,
    param_df,
    q0,
    qlats,
    usgs_df,
    assume_short_ts,
    return_courant,
    waterbodies_df,
    diffusive_parameters=None,
):

    start_time = time.time()
    compute_func = _compute_func_map[compute_func_name]
    if parallel_compute_method == "by-subnetwork-jit-clustered":
        networks_with_subnetworks_ordered_jit = nhd_network.build_subnetworks(
            connections, rconn, subnetwork_target_size
        )
        subnetworks_only_ordered_jit = defaultdict(dict)
        subnetworks = defaultdict(dict)
        for tw, ordered_network in networks_with_subnetworks_ordered_jit.items():
            intw = independent_networks[tw]
            for order, subnet_sets in ordered_network.items():
                subnetworks_only_ordered_jit[order].update(subnet_sets)
                for subn_tw, subnetwork in subnet_sets.items():
                    subnetworks[subn_tw] = {k: intw[k] for k in subnetwork}

        reaches_ordered_bysubntw = defaultdict(dict)
        for order, ordered_subn_dict in subnetworks_only_ordered_jit.items():
            for subn_tw, subnet in ordered_subn_dict.items():
                conn_subn = {k: connections[k] for k in subnet if k in connections}
                rconn_subn = {k: rconn[k] for k in subnet if k in rconn}
                path_func = partial(nhd_network.split_at_junction, rconn_subn)
                reaches_ordered_bysubntw[order][
                    subn_tw
                ] = nhd_network.dfs_decomposition(rconn_subn, path_func)

        cluster_threshold = 0.65  # When a job has a total segment count 65% of the target size, compute it
        # Otherwise, keep adding reaches.

        reaches_ordered_bysubntw_clustered = defaultdict(dict)

        for order in subnetworks_only_ordered_jit:
            cluster = 0
            reaches_ordered_bysubntw_clustered[order][cluster] = {
                "segs": [],
                "upstreams": {},
                "tw": [],
                "subn_reach_list": [],
            }
            for twi, (subn_tw, subn_reach_list) in enumerate(
                reaches_ordered_bysubntw[order].items(), 1
            ):
                segs = list(chain.from_iterable(subn_reach_list))
                reaches_ordered_bysubntw_clustered[order][cluster]["segs"].extend(segs)
                reaches_ordered_bysubntw_clustered[order][cluster]["upstreams"].update(
                    subnetworks[subn_tw]
                )

                reaches_ordered_bysubntw_clustered[order][cluster]["tw"].append(subn_tw)
                reaches_ordered_bysubntw_clustered[order][cluster][
                    "subn_reach_list"
                ].extend(subn_reach_list)

                if (
                    len(reaches_ordered_bysubntw_clustered[order][cluster]["segs"])
                    >= cluster_threshold * subnetwork_target_size
                ) and (
                    twi
                    < len(reaches_ordered_bysubntw[order])
                    # i.e., we haven't reached the end
                    # TODO: perhaps this should be a while condition...
                ):
                    cluster += 1
                    reaches_ordered_bysubntw_clustered[order][cluster] = {
                        "segs": [],
                        "upstreams": {},
                        "tw": [],
                        "subn_reach_list": [],
                    }

        if 1 == 1:
            print("JIT Preprocessing time %s seconds." % (time.time() - start_time))
            print("starting Parallel JIT calculation")

        start_para_time = time.time()
        # if 1 == 1:
        with Parallel(n_jobs=cpu_pool, backend="threading") as parallel:
            results_subn = defaultdict(list)
            flowveldepth_interorder = {}

            for order in range(max(subnetworks_only_ordered_jit.keys()), -1, -1):
                jobs = []
                for cluster, clustered_subns in reaches_ordered_bysubntw_clustered[
                    order
                ].items():
                    segs = clustered_subns["segs"]
                    offnetwork_upstreams = set()
                    segs_set = set(segs)
                    for seg in segs:
                        for us in rconn[seg]:
                            if us not in segs_set:
                                offnetwork_upstreams.add(us)

                    segs.extend(offnetwork_upstreams)
                    param_df_sub = param_df.loc[
                        segs, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
                    ].sort_index()
                    if order < max(subnetworks_only_ordered_jit.keys()):
                        for us_subn_tw in offnetwork_upstreams:
                            subn_tw_sortposition = param_df_sub.index.get_loc(
                                us_subn_tw
                            )
                            flowveldepth_interorder[us_subn_tw][
                                "position_index"
                            ] = subn_tw_sortposition

                    subn_reach_list = clustered_subns["subn_reach_list"]
                    upstreams = clustered_subns["upstreams"]

                    if not usgs_df.empty:
                        usgs_segs = list(usgs_df.index.intersection(param_df_sub.index))
                        nudging_positions_list = param_df_sub.index.get_indexer(
                            usgs_segs
                        )
                        usgs_df_sub = usgs_df.loc[usgs_segs]
                        usgs_df_sub.drop(
                            usgs_df_sub.columns[range(0, 1)], axis=1, inplace=True
                        )
                    else:
                        usgs_df_sub = pd.DataFrame()
                        nudging_positions_list = []

                    qlat_sub = qlats.loc[param_df_sub.index]
                    q0_sub = q0.loc[param_df_sub.index]

                    # At present, in by-subnetwork-jit/jit-clustered, these next two lines
                    # only produce a dummy list, but...
                    # Eventually, the wiring for reservoir simulation needs to be added.
                    subn_reach_type_list = [0 for reaches in subn_reach_list]
                    subn_reach_list_with_type = list(
                        zip(subn_reach_list, subn_reach_type_list)
                    )

                    # results_subn[order].append(
                    #     compute_func(
                    jobs.append(
                        delayed(compute_func)(
                            nts,
                            qts_subdivisions,
                            subn_reach_list_with_type,
                            upstreams,
                            param_df_sub.index.values,
                            param_df_sub.columns.values,
                            param_df_sub.values,
                            q0_sub.values.astype("float32"),
                            qlat_sub.values.astype("float32"),
                            [],  # lake_segs
                            np.empty(
                                shape=(0, 0), dtype="float64"
                            ),  # waterbodies_df_sub.values
                            usgs_df_sub.values.astype("float32"),
                            # flowveldepth_interorder,  # obtain keys and values from this dataset
                            np.array(nudging_positions_list, dtype="int32"),
                            {
                                us: fvd
                                for us, fvd in flowveldepth_interorder.items()
                                if us in offnetwork_upstreams
                            },
                            assume_short_ts,
                            return_courant,
                            diffusive_parameters,
                        )
                    )
                results_subn[order] = parallel(jobs)

                if order > 0:  # This is not needed for the last rank of subnetworks
                    flowveldepth_interorder = {}
                    for ci, (cluster, clustered_subns) in enumerate(
                        reaches_ordered_bysubntw_clustered[order].items()
                    ):
                        for subn_tw in clustered_subns["tw"]:
                            # TODO: This index step is necessary because we sort the segment index
                            # TODO: I think there are a number of ways we could remove the sorting step
                            #       -- the binary search could be replaced with an index based on the known topology
                            flowveldepth_interorder[subn_tw] = {}
                            subn_tw_sortposition = (
                                results_subn[order][ci][0].tolist().index(subn_tw)
                            )
                            flowveldepth_interorder[subn_tw]["results"] = results_subn[
                                order
                            ][ci][1][subn_tw_sortposition]
                            # what will it take to get just the tw FVD values into an array to pass to the next loop?
                            # There will be an empty array initialized at the top of the loop, then re-populated here.
                            # we don't have to bother with populating it after the last group

        results = []
        for order in subnetworks_only_ordered_jit:
            results.extend(results_subn[order])

        if 1 == 1:
            print("PARALLEL TIME %s seconds." % (time.time() - start_para_time))

    elif parallel_compute_method == "by-subnetwork-jit":
        networks_with_subnetworks_ordered_jit = nhd_network.build_subnetworks(
            connections, rconn, subnetwork_target_size
        )
        subnetworks_only_ordered_jit = defaultdict(dict)
        subnetworks = defaultdict(dict)
        for tw, ordered_network in networks_with_subnetworks_ordered_jit.items():
            intw = independent_networks[tw]
            for order, subnet_sets in ordered_network.items():
                subnetworks_only_ordered_jit[order].update(subnet_sets)
                for subn_tw, subnetwork in subnet_sets.items():
                    subnetworks[subn_tw] = {k: intw[k] for k in subnetwork}

        reaches_ordered_bysubntw = defaultdict(dict)
        for order, ordered_subn_dict in subnetworks_only_ordered_jit.items():
            for subn_tw, subnet in ordered_subn_dict.items():
                conn_subn = {k: connections[k] for k in subnet if k in connections}
                rconn_subn = {k: rconn[k] for k in subnet if k in rconn}
                path_func = partial(nhd_network.split_at_junction, rconn_subn)
                reaches_ordered_bysubntw[order][
                    subn_tw
                ] = nhd_network.dfs_decomposition(rconn_subn, path_func)

        if 1 == 1:
            print("JIT Preprocessing time %s seconds." % (time.time() - start_time))
            print("starting Parallel JIT calculation")

        start_para_time = time.time()
        with Parallel(n_jobs=cpu_pool, backend="threading") as parallel:
            results_subn = defaultdict(list)
            flowveldepth_interorder = {}

            for order in range(max(subnetworks_only_ordered_jit.keys()), -1, -1):
                jobs = []
                for twi, (subn_tw, subn_reach_list) in enumerate(
                    reaches_ordered_bysubntw[order].items(), 1
                ):
                    # TODO: Confirm that a list here is best -- we are sorting,
                    # so a set might be sufficient/better
                    segs = list(chain.from_iterable(subn_reach_list))
                    offnetwork_upstreams = set()
                    segs_set = set(segs)
                    for seg in segs:
                        for us in rconn[seg]:
                            if us not in segs_set:
                                offnetwork_upstreams.add(us)

                    segs.extend(offnetwork_upstreams)
                    param_df_sub = param_df.loc[
                        segs, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
                    ].sort_index()
                    if order < max(subnetworks_only_ordered_jit.keys()):
                        for us_subn_tw in offnetwork_upstreams:
                            subn_tw_sortposition = param_df_sub.index.get_loc(
                                us_subn_tw
                            )
                            flowveldepth_interorder[us_subn_tw][
                                "position_index"
                            ] = subn_tw_sortposition

                    if not usgs_df.empty:
                        usgs_segs = list(usgs_df.index.intersection(param_df_sub.index))
                        nudging_positions_list = param_df_sub.index.get_indexer(
                            usgs_segs
                        )
                        usgs_df_sub = usgs_df.loc[usgs_segs]
                        usgs_df_sub.drop(
                            usgs_df_sub.columns[range(0, 1)], axis=1, inplace=True
                        )
                    else:
                        usgs_df_sub = pd.DataFrame()
                        nudging_positions_list = []

                    qlat_sub = qlats.loc[param_df_sub.index]
                    q0_sub = q0.loc[param_df_sub.index]

                    # At present, in by-subnetwork-jit/jit-clustered, these next two lines
                    # only produce a dummy list, but...
                    # Eventually, the wiring for reservoir simulation needs to be added.
                    subn_reach_type_list = [0 for reaches in subn_reach_list]
                    subn_reach_list_with_type = list(
                        zip(subn_reach_list, subn_reach_type_list)
                    )

                    jobs.append(
                        delayed(compute_func)(
                            nts,
                            qts_subdivisions,
                            subn_reach_list_with_type,
                            subnetworks[subn_tw],
                            param_df_sub.index.values,
                            param_df_sub.columns.values,
                            param_df_sub.values,
                            q0_sub.values.astype("float32"),
                            qlat_sub.values.astype("float32"),
                            [],  # lake_segs
                            np.empty(
                                shape=(0, 0), dtype="float64"
                            ),  # waterbodies_df_sub.values
                            usgs_df_sub.values.astype("float32"),
                            # flowveldepth_interorder,  # obtain keys and values from this dataset
                            np.array(nudging_positions_list, dtype="int32"),
                            {
                                us: fvd
                                for us, fvd in flowveldepth_interorder.items()
                                if us in offnetwork_upstreams
                            },
                            assume_short_ts,
                            return_courant,
                            diffusive_parameters,
                        )
                    )

                results_subn[order] = parallel(jobs)

                if order > 0:  # This is not needed for the last rank of subnetworks
                    flowveldepth_interorder = {}
                    for twi, subn_tw in enumerate(reaches_ordered_bysubntw[order]):
                        # TODO: This index step is necessary because we sort the segment index
                        # TODO: I think there are a number of ways we could remove the sorting step
                        #       -- the binary search could be replaced with an index based on the known topology
                        flowveldepth_interorder[subn_tw] = {}
                        subn_tw_sortposition = (
                            results_subn[order][twi][0].tolist().index(subn_tw)
                        )
                        flowveldepth_interorder[subn_tw]["results"] = results_subn[
                            order
                        ][twi][1][subn_tw_sortposition]
                        # what will it take to get just the tw FVD values into an array to pass to the next loop?
                        # There will be an empty array initialized at the top of the loop, then re-populated here.
                        # we don't have to bother with populating it after the last group

        results = []
        for order in subnetworks_only_ordered_jit:
            results.extend(results_subn[order])

        if 1 == 1:
            print("PARALLEL TIME %s seconds." % (time.time() - start_para_time))

    elif parallel_compute_method == "by-network":
        with Parallel(n_jobs=cpu_pool, backend="threading") as parallel:
            jobs = []
            for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
                # The X_sub lines use SEGS...
                # which is now invalid with the wbodies included.
                # So we define "common_segs" to identify regular routing segments
                # and wbodies_segs for the waterbody reaches/segments
                segs = list(chain.from_iterable(reach_list))
                common_segs = param_df.index.intersection(segs)
                # Assumes everything else is a waterbody...
                wbodies_segs = set(segs).symmetric_difference(common_segs)

                # If waterbody parameters exist
                if not waterbodies_df.empty:

                    lake_segs = list(waterbodies_df.index.intersection(segs))

                    waterbodies_df_sub = waterbodies_df.loc[
                        lake_segs,
                        [
                            "LkArea",
                            "LkMxE",
                            "OrificeA",
                            "OrificeC",
                            "OrificeE",
                            "WeirC",
                            "WeirE",
                            "WeirL",
                            "ifd",
                            "qd0",
                            "h0",
                        ],
                    ]

                else:
                    lake_segs = []
                    waterbodies_df_sub = pd.DataFrame()

                param_df_sub = param_df.loc[
                    common_segs,
                    ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"],
                ].sort_index()

                if not usgs_df.empty:
                    usgs_segs = list(usgs_df.index.intersection(param_df_sub.index))
                    nudging_positions_list = param_df_sub.index.get_indexer(usgs_segs)
                    usgs_df_sub = usgs_df.loc[usgs_segs]
                    usgs_df_sub.drop(
                        usgs_df_sub.columns[range(0, 1)], axis=1, inplace=True
                    )
                else:
                    usgs_df_sub = pd.DataFrame()
                    nudging_positions_list = []

                reaches_list_with_type = []

                for reaches in reach_list:
                    if set(reaches) & wbodies_segs:
                        reach_type = 1  # type 1 for waterbody/lake
                    else:
                        reach_type = 0  # type 0 for reach

                    reach_and_type_tuple = (reaches, reach_type)

                    reaches_list_with_type.append(reach_and_type_tuple)

                # qlat_sub = qlats.loc[common_segs].sort_index()
                # q0_sub = q0.loc[common_segs].sort_index()
                qlat_sub = qlats.loc[param_df_sub.index]
                q0_sub = q0.loc[param_df_sub.index]

                param_df_sub = param_df_sub.reindex(
                    param_df_sub.index.tolist() + lake_segs
                ).sort_index()
                qlat_sub = qlat_sub.reindex(param_df_sub.index)
                q0_sub = q0_sub.reindex(param_df_sub.index)

                jobs.append(
                    delayed(compute_func)(
                        nts,
                        qts_subdivisions,
                        reaches_list_with_type,
                        independent_networks[tw],
                        param_df_sub.index.values.astype("int64"),
                        param_df_sub.columns.values,
                        param_df_sub.values,
                        q0_sub.values.astype("float32"),
                        qlat_sub.values.astype("float32"),
                        lake_segs,
                        waterbodies_df_sub.values,
                        usgs_df_sub.values.astype("float32"),
                        np.array(nudging_positions_list, dtype="int32"),
                        {},
                        assume_short_ts,
                        return_courant,
                        diffusive_parameters,
                    )
                )
            results = parallel(jobs)

    else:  # Execute in serial
        results = []
        for twi, (tw, reach_list) in enumerate(reaches_bytw.items(), 1):
            # The X_sub lines use SEGS...
            # which becomes invalid with the wbodies included.
            # So we define "common_segs" to identify regular routing segments
            # and wbodies_segs for the waterbody reaches/segments
            segs = list(chain.from_iterable(reach_list))
            common_segs = param_df.index.intersection(segs)
            # Assumes everything else is a waterbody...
            wbodies_segs = set(segs).symmetric_difference(common_segs)

            # If waterbody parameters exist
            if not waterbodies_df.empty:

                lake_segs = list(waterbodies_df.index.intersection(segs))

                waterbodies_df_sub = waterbodies_df.loc[
                    lake_segs,
                    [
                        "LkArea",
                        "LkMxE",
                        "OrificeA",
                        "OrificeC",
                        "OrificeE",
                        "WeirC",
                        "WeirE",
                        "WeirL",
                        "ifd",
                        "qd0",
                        "h0",
                    ],
                ]

            else:
                lake_segs = []
                waterbodies_df_sub = pd.DataFrame()

            param_df_sub = param_df.loc[
                common_segs,
                ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
            ].sort_index()

            if not usgs_df.empty:
                usgs_segs = list(usgs_df.index.intersection(param_df_sub.index))
                nudging_positions_list = param_df_sub.index.get_indexer(usgs_segs)
                usgs_df_sub = usgs_df.loc[usgs_segs]
                usgs_df_sub.drop(usgs_df_sub.columns[range(0, 1)], axis=1, inplace=True)
            else:
                usgs_df_sub = pd.DataFrame()
                nudging_positions_list = []

            # qlat_sub = qlats.loc[common_segs].sort_index()
            # q0_sub = q0.loc[common_segs].sort_index()
            qlat_sub = qlats.loc[param_df_sub.index]
            q0_sub = q0.loc[param_df_sub.index]

            param_df_sub = param_df_sub.reindex(
                param_df_sub.index.tolist() + lake_segs
            ).sort_index()
            qlat_sub = qlat_sub.reindex(param_df_sub.index)
            q0_sub = q0_sub.reindex(param_df_sub.index)

            reach_type_list = [
                1 if (set(reaches) & wbodies_segs) else 0 for reaches in reach_list
            ]
            reaches_list_with_type = list(zip(reach_list, reach_type_list))
            """
            reaches_list_with_type = []

            for reaches in reach_list:
                if (set(reaches) & wbodies_segs):
                    reach_type = 1 # type 1 for waterbody/lake
                else:
                    reach_type = 0 # type 0 for reach

                reach_and_type_tuple = (reaches, reach_type)

                reaches_list_with_type.append(reach_and_type_tuple)
            """

            results.append(
                compute_func(
                    nts,
                    qts_subdivisions,
                    reaches_list_with_type,
                    independent_networks[tw],
                    param_df_sub.index.values.astype("int64"),
                    param_df_sub.columns.values,
                    param_df_sub.values,
                    q0_sub.values.astype("float32"),
                    qlat_sub.values.astype("float32"),
                    lake_segs,
                    waterbodies_df_sub.values,
                    usgs_df_sub.values.astype("float32"),
                    np.array(nudging_positions_list, dtype="int32"),
                    {},
                    assume_short_ts,
                    return_courant,
                    diffusive_parameters,
                )
            )

    return results