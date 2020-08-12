import networkbuilder as networkbuilder
import os
import traceback
import geopandas as gpd
import pandas as pd
import zipfile
import xarray as xr
import network_dl
import json


def get_geo_file_table_rows(
    geo_file_path=None,
    data_link=None,
    layer_string=None,
    driver_string=None,
    verbose=False,
    debuglevel=0,
):

    if not os.path.exists(geo_file_path):
        filename = os.path.basename(geo_file_path)
        msg = ""
        msg = msg + f"\nTarget input file not found on file system here:"
        msg = msg + f"\n{geo_file_path}"
        msg = msg + f"\n"
        msg = msg + f"\nThis routine will attempt to download the file from here:"
        msg = msg + f"\n{data_link}"
        msg = (
            msg
            + f"\nYou should not need to repeat this step once the file is downloaded."
        )
        msg = msg + f"\n"
        msg = (
            msg
            + f"\nIf you wish to manually download the file (and compress it), please use the following commands:"
        )
        msg = msg + f'\n# export ROUTELINK="{filename}"'
        msg = (
            msg
            + f"\n# wget -c https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/nwm.v2.0.2/parm/domain/$ROUTELINK"
        )
        msg = (
            msg
            + f"\n# export $ROUTELINK_UNCOMPRESSED=${{ROUTELINK/\.nc/_uncompressed.nc}}"
        )
        msg = msg + f"\n# mv $ROUTELINK $ROUTELINK_UNCOMPRESSED"
        msg = msg + f"\n# nccopy -d1 -s $ROUTELINK_UNCOMPRESSED $ROUTELINK"

        """
        Full CONUS route-link file not found on file system.
        This routine will attempt to download the file. 

        If you wish to manually download the file, please use the following commands:

        > export ROUTELINK="RouteLink_NHDPLUS.nc"
        > wget -c https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/nwm.v2.0.2/parm/domain/$ROUTELINK
        > nccopy -d1 -s $ROUTELINK ${ROUTELINK/\.nc/_compressed.nc}
        """
        print(msg)

        network_dl.download(geo_file_path, data_link)

    # NOTE: these methods can lose the "connections" and "rows" arguments when
    # implemented as class methods where those arrays are members of the class.
    # TODO: Improve the error handling here for a corrupt input file
    if debuglevel <= -1:
        print(
            f"reading -- dataset: {geo_file_path}; layer: {layer_string}; driver: {driver_string}"
        )
    if driver_string == "NetCDF":  # Use Xarray to read a netcdf table
        try:
            geo_file = xr.open_dataset(geo_file_path)
            geo_file_rows = (geo_file.to_dataframe()).values
        except Exception as e:
            print(e)
            if debuglevel <= -1:
                traceback.print_exc()
        # The xarray method for NetCDFs was implemented after the geopandas method for
        # GIS source files. It's possible (probable?) that we are doing something
        # inefficient by converting away from the Pandas dataframe.
        # TODO: Check the optimal use of the Pandas dataframe
    elif driver_string == "csv":  # Use Pandas to read zipped csv/txt
        try:
            HEADER = None  # TODO: standardize the mask format or add some logic to handle headers or other variations in format
            geo_file = pd.read_csv(geo_file_path, header=HEADER)
        except Exception as e:
            print(e)
            if debuglevel <= -1:
                traceback.print_exc()
        geo_file_rows = geo_file.to_numpy()
    elif driver_string == "zip":  # Use Pandas to read zipped csv/txt
        try:
            with zipfile.ZipFile(geo_file_path, "r") as zcsv:
                with zcsv.open(layer_string) as csv:
                    geo_file = pd.read_csv(csv)
        except Exception as e:
            print(e)
            if debuglevel <= -1:
                traceback.print_exc()
        geo_file_rows = geo_file.to_numpy()
    else:  # Read Shapefiles, Geodatabases with Geopandas/fiona
        try:
            geo_file = gpd.read_file(
                geo_file_path, driver=driver_string, layer=layer_string
            )
            geo_file_rows = geo_file.to_numpy()
        except Exception as e:
            print(e)
            if debuglevel <= -1:
                traceback.print_exc()
        if debuglevel <= -2:
            try:
                geo_file.plot()
            except Exception as e:
                print(r"cannot plot geofile (not necessarily a problem)")
                traceback.print_exc()
    if debuglevel <= -1:
        # official docs here:
        # https://pandas.pydata.org/docs/user_guide/options.html
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", -1)
        print(geo_file.head())  # Preview the first 5 lines of the loaded data

    return geo_file_rows


# TODO: Give this function a more appropriate general name (it does more that build connections)
def build_connections_object(
    geo_file_rows=None,
    mask_set=None,
    key_col=None,
    downstream_col=None,
    length_col=None,
    terminal_code=None,
    waterbody_col=None,
    waterbody_null_code=None,
    verbose=False,
    debuglevel=0,
):
    (connections) = networkbuilder.get_down_connections(
        rows=geo_file_rows,
        mask_set=mask_set,
        key_col=key_col,
        downstream_col=downstream_col,
        length_col=length_col,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    (
        all_keys,
        ref_keys,
        headwater_keys,
        terminal_keys,
        terminal_ref_keys,
        circular_keys,
    ) = networkbuilder.determine_keys(
        connections=connections
        # , key_col = key_col
        # , downstream_col = downstream_col
        ,
        terminal_code=terminal_code,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    (
        junction_keys,
        confluence_segment_set,
        visited_keys,
        visited_terminal_keys,
        junction_count,
    ) = networkbuilder.get_up_connections(
        connections=connections,
        terminal_code=terminal_code,
        headwater_keys=headwater_keys,
        terminal_keys=terminal_keys,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    waterbody_set = None
    waterbody_segments = None
    waterbody_outlet_set = None
    waterbody_upstreams_set = None
    waterbody_downstream_set = None

    # TODO: Set/pass/identify a proper flag value
    if waterbody_col is not None:
        (
            waterbody_set,
            waterbody_segments,
            waterbody_outlet_set,
            waterbody_upstreams_set,
            waterbody_downstream_set,
        ) = networkbuilder.get_waterbody_segments(
            connections=connections,
            terminal_code=terminal_code,
            waterbody_col=waterbody_col,
            waterbody_null_code=waterbody_null_code,
            verbose=verbose,
            debuglevel=debuglevel
            # , debuglevel = -3
        )

    # TODO: change names to reflect type set or dict
    return (
        connections,
        all_keys,
        ref_keys,
        headwater_keys,
        terminal_keys,
        terminal_ref_keys,
        circular_keys,
        junction_keys,
        visited_keys,
        visited_terminal_keys,
        junction_count,
        confluence_segment_set,
        waterbody_set,
        waterbody_segments,
        waterbody_outlet_set,
        waterbody_upstreams_set,
        waterbody_downstream_set,
    )


def do_connections(
    geo_file_path=None,
    data_link=None,
    title_string=None,
    layer_string=None,
    driver_string=None,
    key_col=None,
    downstream_col=None,
    length_col=None,
    terminal_code=None,
    waterbody_col=None,
    waterbody_null_code=None,
    mask_file_path=None,
    mask_driver_string=None,
    mask_layer_string=None,
    mask_key_col=None,
    verbose=False,
    debuglevel=0,
):

    if verbose:
        print(title_string)
    geo_file_rows = get_geo_file_table_rows(
        geo_file_path=geo_file_path,
        data_link=data_link,
        layer_string=layer_string,
        driver_string=driver_string,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    if debuglevel <= -1:
        print(f"MASK: {mask_file_path}")
    if mask_file_path:
        mask_file_rows = get_geo_file_table_rows(
            geo_file_path=mask_file_path,
            layer_string=mask_layer_string,
            driver_string=mask_driver_string,
            verbose=verbose,
            debuglevel=debuglevel,
        )
        # TODO: make mask dict with additional attributes, e.g., names
        mask_set = {row[mask_key_col] for row in mask_file_rows}
    else:
        mask_set = {row[key_col] for row in geo_file_rows}

    return build_connections_object(
        geo_file_rows=geo_file_rows,
        mask_set=mask_set,
        key_col=key_col,
        downstream_col=downstream_col,
        length_col=length_col,
        terminal_code=terminal_code,
        waterbody_col=waterbody_col,
        waterbody_null_code=waterbody_null_code,
        verbose=verbose,
        debuglevel=debuglevel,
    )

    # return connections, all_keys, ref_keys, headwater_keys \
    #     , terminal_keys, terminal_ref_keys \
    #     , circular_keys, junction_keys \
    #     , visited_keys, visited_terminal_keys \
    #     , junction_count


def get_nhd_connections(supernetwork_data={}, debuglevel=0, verbose=False):
    if "waterbody_col" not in supernetwork_data:
        supernetwork_data.update({"waterbody_col": None})
        supernetwork_data.update({"waterbody_null_code": None})

    if "mask_file_path" not in supernetwork_data:
        # TODO: doing things this way may mean we are reading the same file twice -- fix this [maybe] by implementing an overloaded return
        supernetwork_data.update({"mask_file_path": None})
        supernetwork_data.update({"mask_layer_string": None})
        supernetwork_data.update({"mask_driver_string": None})
        supernetwork_data.update({"mask_key_col": None})

    if "data_link" not in supernetwork_data:
        supernetwork_data.update({"data_link": None})

    return do_connections(
        geo_file_path=supernetwork_data["geo_file_path"],
        data_link=supernetwork_data["data_link"],
        key_col=supernetwork_data["key_col"],
        downstream_col=supernetwork_data["downstream_col"],
        length_col=supernetwork_data["length_col"],
        terminal_code=supernetwork_data["terminal_code"],
        waterbody_col=supernetwork_data["waterbody_col"],
        waterbody_null_code=supernetwork_data["waterbody_null_code"],
        title_string=supernetwork_data["title_string"],
        driver_string=supernetwork_data["driver_string"],
        layer_string=supernetwork_data["layer_string"],
        mask_file_path=supernetwork_data["mask_file_path"],
        mask_layer_string=supernetwork_data["mask_layer_string"],
        mask_driver_string=supernetwork_data["mask_driver_string"],
        mask_key_col=supernetwork_data["mask_key_col"],
        debuglevel=debuglevel,
        verbose=verbose,
    )


def set_supernetwork_data(
    supernetwork="", geo_input_folder=None, verbose=True, debuglevel=0
):

    # The following datasets are extracts from the feature datasets available
    # from https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/web/data_tools/
    # the CONUS_ge5 and Brazos_LowerColorado_ge5 datasets are included
    # in the github test folder

    supernetwork_options = {
        "Pocono_TEST1",
        "Pocono_TEST2",
        "LowerColorado_Conchos_FULL_RES",
        "Brazos_LowerColorado_ge5",
        "Brazos_LowerColorado_FULL_RES",
        "Brazos_LowerColorado_Named_Streams",
        "CONUS_ge5",
        "Mainstems_CONUS",
        "CONUS_Named_Streams",
        "CONUS_FULL_RES_v20",
        "custom",
    }
    if supernetwork not in supernetwork_options:
        print(
            "Note: please call function with supernetworks set to one of the following:"
        )
        for s in supernetwork_options:
            print(f"'{s}'")
        exit()
    elif supernetwork == "Pocono_TEST1":
        return {
            "geo_file_path": os.path.join(
                geo_input_folder, r"PoconoSampleData1", r"PoconoSampleRouteLink1.shp"
            ),
            "key_col": 18,
            "downstream_col": 23,
            "length_col": 5,
            "manningn_col": 20,
            "manningncc_col": 21,
            "slope_col": 10,
            "bottomwidth_col": 2,
            "topwidth_col": 11,
            "topwidthcc_col": 12,
            "waterbody_col": 8,
            "waterbody_null_code": -9999,
            "MusK_col": 7,
            "MusX_col": 8,
            "ChSlp_col": 12,
            "terminal_code": 0,
            "title_string": "Pocono Test Example",
            "driver_string": "ESRI Shapefile",
            "layer_string": 0,
            "waterbody_parameter_file_type": "Level_Pool",
            "waterbody_parameters": {
                "level_pool_waterbody_parameter_file_path": os.path.join(
                    geo_input_folder, "NWM_2.1_Sample_Datasets", "LAKEPARM_CONUS.nc"
                ),
                "level_pool_waterbody_area": "LkArea",  # area of reservoir
                "level_pool_weir_elevation": "WeirE",
                "level_pool_waterbody_max_elevation": "LkMxE",
                "level_pool_outfall_weir_coefficient": "WeirC",
                "level_pool_outfall_weir_length": "WeirL",
                "level_pool_overall_dam_length": "DamL",
                "level_pool_orifice_elevation": "OrificeE",
                "level_pool_orifice_coefficient": "OrificeC",
                "level_pool_orifice_area": "OrificeA",
            },
        }

    elif supernetwork == "Pocono_TEST2":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "Pocono Test 2 Example",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder,
                    r"Channels",
                    r"masks",
                    r"PoconoRouteLink_TEST2_nwm_mc.txt",
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

        # return {
        #'geo_file_path' : os.path.join(geo_input_folder
    # , r'PoconoSampleData2'
    # , r'PoconoRouteLink_testsamp1_nwm_mc.shp')
    # , 'key_col' : 18
    # , 'downstream_col' : 23
    # , 'length_col' : 5
    # , 'manningn_col' : 20
    # , 'manningncc_col' : 21
    # , 'slope_col' : 10
    # , 'bottomwidth_col' : 2
    # , 'topwidth_col' : 11
    # , 'topwidthcc_col' : 12
    # , 'MusK_col' : 6
    # , 'MusX_col' : 7
    # , 'ChSlp_col' : 3
    # , 'terminal_code' : 0
    # , 'title_string' : 'Pocono Test 2 Example'
    # , 'driver_string' : 'ESRI Shapefile'
    # , 'layer_string' : 0
    # }

    elif supernetwork == "LowerColorado_Conchos_FULL_RES":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "NHD 2.0 Conchos Basin of the LowerColorado River",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder,
                    r"Channels",
                    r"masks",
                    r"LowerColorado_Conchos_FULL_RES.txt",
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

    elif supernetwork == "Brazos_LowerColorado_ge5":
        return {
            "geo_file_path": os.path.join(
                geo_input_folder, r"Channels", r"NHD_BrazosLowerColorado_Channels.shp"
            ),
            "key_col": 2,
            "downstream_col": 7,
            "length_col": 6,
            "manningn_col": 11,
            "slope_col": 10,
            "bottomwidth_col": 12,
            "MusK_col": 7,
            "MusX_col": 8,
            "ChSlp_col": 13,
            "terminal_code": 0,
            "title_string": "NHD Subset including Brazos + Lower Colorado\nNHD stream orders 5 and greater",
            "driver_string": "ESRI Shapefile",
            "layer_string": 0,
        }

    elif supernetwork == "Brazos_LowerColorado_FULL_RES":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "NHD 2.0 Brazos and LowerColorado Basins",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder,
                    r"Channels",
                    r"masks",
                    r"Brazos_LowerColorado_FULL_RES.txt",
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

    elif supernetwork == "Brazos_LowerColorado_Named_Streams":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "NHD 2.0 GNIS labeled streams in the Brazos and LowerColorado Basins",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder,
                    r"Channels",
                    r"masks",
                    r"Brazos_LowerColorado_Named_Streams.csv",
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

    elif supernetwork == "CONUS_ge5":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "NHD CONUS Order 5 and Greater",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder, r"Channels", r"masks", r"CONUS_ge5.txt"
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

    elif supernetwork == "Mainstems_CONUS":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "CONUS 'Mainstems' (Channels below gages and AHPS prediction points)",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder, r"Channels", r"masks", r"conus_Mainstem_links.txt"
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

        # return {
        #     "geo_file_path": os.path.join(
        #         geo_input_folder, r"Channels", r"conus_routeLink_subset.nc"
        #     ),
        #     "key_col": 0,
        #     "downstream_col": 2,
        #     "length_col": 10,
        #     "manningn_col": 11,
        #     "manningncc_col": 20,
        #     "slope_col": 12,
        #     "bottomwidth_col": 14,
        #     "topwidth_col": 22,
        #     "topwidthcc_col": 21,
        #     "waterbody_col": 15,
        #     "waterbody_null_code": -9999,
        #     "MusK_col": 8,
        #     "MusX_col": 9,
        #     "ChSlp_col": 13,
        #     "terminal_code": 0,
        #     "title_string": 'CONUS "Mainstem"',
        #     "driver_string": "NetCDF",
        #     "layer_string": 0,
        # }

    elif supernetwork == "CONUS_Named_Streams":
        dict = set_supernetwork_data(
            supernetwork="CONUS_FULL_RES_v20", geo_input_folder=geo_input_folder
        )
        dict.update(
            {
                "title_string": "CONUS NWM v2.0 only GNIS labeled streams",  # overwrites other title...
                "mask_file_path": os.path.join(
                    geo_input_folder,
                    r"Channels",
                    r"masks",
                    r"nwm_reaches_conus_v21_wgnis_name.csv",
                ),
                "mask_driver_string": r"csv",
                "mask_layer_string": r"",
                "mask_key_col": 0,
                "mask_name_col": 1,  # TODO: Not used yet.
            }
        )
        return dict

    elif supernetwork == "CONUS_FULL_RES_v20":

        ROUTELINK = r"RouteLink_NHDPLUS"
        ModelVer = r"nwm.v2.0.2"
        ext = r"nc"
        sep = r"."

        return {
            "geo_file_path": os.path.join(
                geo_input_folder, r"Channels", sep.join([ROUTELINK, ModelVer, ext])
            ),
            "data_link": f"https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/{ModelVer}/parm/domain/{ROUTELINK}{sep}{ext}",
            "key_col": 0,
            "downstream_col": 2,
            "length_col": 10,
            "manningn_col": 11,
            "manningncc_col": 20,
            "slope_col": 12,
            "bottomwidth_col": 14,
            "topwidth_col": 22,
            "topwidthcc_col": 21,
            "waterbody_col": 15,
            "waterbody_null_code": -9999,
            "MusK_col": 8,
            "MusX_col": 9,
            "ChSlp_col": 13,
            "terminal_code": 0,
            "title_string": "CONUS Full Resolution NWM v2.0",
            "driver_string": "NetCDF",
            "layer_string": 0,
            "waterbody_parameter_file_type": "Level_Pool",
            "ql_input_folder": r"/home/APD/inland_hydraulics/wrf-hydro-run/OUTPUTS",
            "ql_files_tail": "/*.CHRTOUT_DOMAIN1",
            "time_string": "2020-03-19_18:00_DOMAIN1",
            "channel_initial_states_file": r"/home/APD/inland_hydraulics/wrf-hydro-run/restart/HYDRO_RST.",
            "initial_states_waterbody_ID_crosswalk_file": r"/home/APD/inland_hydraulics/wrf-hydro-run/DOMAIN/waterbody_subset_ID_crosswalk.csv",
            "waterbody_parameters": {
                "level_pool_waterbody_parameter_file_path": os.path.join(
                    geo_input_folder, "NWM_2.1_Sample_Datasets", "LAKEPARM_CONUS.nc"
                ),
                "level_pool_waterbody_area": "LkArea",  # area of reservoir
                "level_pool_weir_elevation": "WeirE",
                "level_pool_waterbody_max_elevation": "LkMxE",
                "level_pool_outfall_weir_coefficient": "WeirC",
                "level_pool_outfall_weir_length": "WeirL",
                "level_pool_overall_dam_length": "DamL",
                "level_pool_orifice_elevation": "OrificeE",
                "level_pool_orifice_coefficient": "OrificeC",
                "level_pool_orifice_area": "OrificeA",
            },
        }

    elif supernetwork == "custom":
        custominput = os.path.join(geo_input_folder)
        with open(custominput) as json_file:
            data = json.load(json_file)
            # TODO: add error trapping for potentially missing files
        return data


# TODO: confirm that this function is not used, and if so, consider removing it
def set_networks(supernetwork="", geo_input_folder=None, verbose=True, debuglevel=0):

    supernetwork_data = set_supernetwork_data(
        supernetwork=supernetwork, geo_input_folder=geo_input_folder
    )
    supernetwork_values = get_nhd_connections(
        supernetwork_data=supernetwork_data, verbose=verbose, debuglevel=debuglevel
    )
    return supernetwork_data, supernetwork_values


def read_waterbody_df(parm_file, lake_id_mask=None):
    ds = xr.open_dataset(parm_file)
    df1 = ds.to_dataframe().set_index("lake_id")
    if lake_id_mask:
        df1 = df1.loc[lake_id_mask, :]
    return df1


def get_ql_from_wrf_hydro(ql_files):
    li = []

    for filename in ql_files:
        ds = xr.open_dataset(filename)
        df1 = ds.to_dataframe()

        li.append(df1)

    frame = pd.concat(li, axis=0, ignore_index=False)
    mod = frame.reset_index()
    ql = mod.pivot(index="station_id", columns="time", values="q_lateral")

    return ql


def get_stream_restart_from_wrf_hydro(
    channel_initial_states_file, initial_states_stream_ID_crosswalk_file
):

    ds = xr.open_dataset(initial_states_stream_ID_crosswalk_file)

    frame = ds.to_dataframe()
    mod = frame.reset_index()

    mod = mod.set_index("station")
    mod = mod[:109223]

    ds2 = xr.open_dataset(channel_initial_states_file)
    qdf = ds2.to_dataframe()
    qdf = qdf.reset_index()
    qdf = qdf.set_index(["links"])
    qdf = qdf[:109223]

    mod = mod.join(qdf)
    mod = mod.drop(
        columns=(
            [
                "time",
                "streamflow",
                "nudge",
                "q_lateral",
                "velocity",
                "qSfcLatRunoff",
                "qBucket",
                "lakes",
                "resht",
                "qlakeo",
            ]
        )
    )
    mod = mod.reset_index()
    mod = mod.set_index(["station_id"])
    q_initial_states = mod

    return q_initial_states


def get_reservoir_restart_from_wrf_hydro(
    waterbody_intial_states_file, initial_states_waterbody_ID_crosswalk_file
):
    # read initial states from r&r output
    # steps to recreate crosswalk csv
    """import xarray as xr
    import pandas as pd
    ds = xr.open_dataset("/home/APD/inland_hydraulics/wrf-hydro-run/DOMAIN/routeLink_subset.nc")
    df2 = ds.to_dataframe()
    df2 = df2.loc[df2['NHDWaterbodyComID']!=-9999]
    unique_WB = df2.NHDWaterbodyComID.unique()
    unique_WB_df = pd.DataFrame(unique_WB,columns=['Waterbody'])
    unique_WB_df.to_csv("/home/APD/inland_hydraulics/wrf-hydro-run/DOMAIN/Waterbody_ID_crosswalk.csv")"""
    ds2 = xr.open_dataset(waterbody_intial_states_file)
    resdf = ds2.to_dataframe()
    resdf = resdf.reset_index()
    resdf = resdf.set_index(["links"])
    resdf = resdf.drop(columns=(["qlink1", "qlink2"]))
    resdf = resdf.loc[0]
    resdf = resdf.reset_index()
    resdf = resdf.set_index(["lakes"])
    resdf = resdf.drop(columns=(["links"]))
    ds = pd.read_csv(initial_states_waterbody_ID_crosswalk_file)
    resdf = resdf.join(ds)
    resdf = resdf.drop(columns=(["Unnamed: 0"]))
    resdf = resdf.reset_index()
    resdf = resdf.set_index(["Waterbody"])
    init_waterbody_states = resdf
    return init_waterbody_states
