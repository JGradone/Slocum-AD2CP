"""
Glider data backends for Slocum-AD2CP.

The AD2CP processing chain needs only a handful of scalars from the glider for
each segment: the time bounds of the segment, the depth-averaged current (DAC),
the local magnetic variation, the endpoint positions, and the maximum depth
reached.  Historically those came from a single source, an ERDDAP tabledap
dataset read by :func:`slocum_ad2cp.get_erddap_dataset`, and the per-segment
scalars were pulled out inline inside the processing notebook.

This module lifts that extraction into one place and puts two interchangeable
backends behind it:

``erddap``
    The original path.  Requires the deployment to be served by an ERDDAP
    server.

``dbd``
    Reads the raw Dinkum binary files (dbd/ebd and their compressed dcd/ecd
    counterparts) straight off the glider's flight and science cards using
    ``dbdreader``.  Works before a deployment is published, exposes sensors that
    ERDDAP datasets often omit, and gives exact segment boundaries because one
    binary file is one segment.

Both backends return the same *segment table*: a :class:`pandas.DataFrame` with
one row per glider segment and the columns listed in :data:`SEGMENT_COLUMNS`.
Downstream code consumes the segment table and therefore does not care which
backend produced it.

Written by Joe Gradone.  dbd backend added 2026.
"""

import glob
import os

import numpy as np
import pandas as pd

from .make_dataset import mag_var_correction


##################################################################################################
# The contract shared by every backend
##################################################################################################

SEGMENT_COLUMNS = [
    "source_file",     # segment identifier (ERDDAP source_file, or dbd basename)
    "start_time",      # first record in the segment, tz-naive UTC
    "end_time",        # last record in the segment, tz-naive UTC
    "start_lat",       # degrees north
    "start_lon",       # degrees east
    "end_lat",
    "end_lon",
    "mid_lat",         # midpoint of start/end, convenience for plotting
    "mid_lon",
    "u_dac",           # eastward DAC, corrected for magnetic variation [m/s]
    "v_dac",           # northward DAC, corrected for magnetic variation [m/s]
    "u_dac_raw",       # DAC as logged by the glider, uncorrected [m/s]
    "v_dac_raw",
    "u_dac_final",     # m_final_water_vx as logged, uncorrected [m/s]
    "v_dac_final",
    "mag_var_deg",     # magnetic variation used for the correction [degrees]
    "max_depth",       # deepest record in the segment [m]
    "n_records",       # number of depth records in the segment
    "dac_time",        # time the DAC value used was logged, tz-naive UTC
]

## Flight-computer parameters the dbd backend reads.  All confirmed present in
## ru37's dbd/dcd files; anything missing is simply returned as all-NaN.
FLIGHT_PARAMS = [
    "m_depth",
    "m_water_vx",
    "m_water_vy",
    "m_final_water_vx",
    "m_final_water_vy",
    "m_gps_mag_var",
    "m_gps_lat",
    "m_gps_lon",
    "m_lat",
    "m_lon",
    "m_heading",
    "m_pitch",
    "m_roll",
]

## Variables requested from ERDDAP by the erddap backend.  Order matters: the
## notebook workflow assigns these names positionally onto the returned frame.
ERDDAP_VARIABLES = [
    "depth",
    "latitude",
    "longitude",
    "time",
    "source_file",
    "m_water_vx",
    "m_water_vy",
    "m_heading",
    "m_gps_mag_var",
]


##################################################################################################

def _empty_segment_row(source_file):
    """An all-NaN segment row, so every code path returns the full schema."""
    row = {c: np.nan for c in SEGMENT_COLUMNS}
    row["source_file"] = source_file
    row["n_records"] = 0
    return row


def _pick_dac(t_wv, u_wv, v_wv, t_final, u_final, v_final, dac_source):
    """
    Choose which logged depth-averaged current represents this segment.

    The flight computer logs ``m_water_vx``/``m_water_vy`` twice per dive
    segment and ``m_final_water_vx``/``m_final_water_vy`` once.  The value
    written at the surfacing that *closes* the segment is that segment's
    estimate; a value written near file open belongs to the previous segment.

    Parameters
    ----------
    t_wv, u_wv, v_wv : ndarray
        Times and values of m_water_vx / m_water_vy within the segment.
    t_final, u_final, v_final : ndarray
        Times and values of m_final_water_vx / m_final_water_vy.
    dac_source : {'last', 'final', 'first'}
        'last'  - last finite m_water_vx/vy (default; matches the behaviour of
                  the original ERDDAP workflow and lands on the closing
                  surfacing in every segment checked).
        'final' - the single m_final_water_vx/vy value.
        'first' - first finite m_water_vx/vy, for diagnostics.

    Returns
    -------
    (u, v, t) : tuple
        Uncorrected DAC components as floats, and the time the value was logged
        in whatever form the caller supplied it.  All NaN if the segment
        carries no usable estimate.
    """
    if dac_source == "final":
        t_src, u_src, v_src, take_last = t_final, u_final, v_final, True
    elif dac_source in ("last", "first"):
        t_src, u_src, v_src = t_wv, u_wv, v_wv
        take_last = dac_source == "last"
    else:
        raise ValueError("dac_source must be one of 'last', 'final', 'first'")

    good = np.isfinite(u_src) & np.isfinite(v_src)
    if not good.any():
        return np.nan, np.nan, np.nan

    idx = np.where(good)[0][-1 if take_last else 0]
    ## The time is returned untouched: the erddap backend passes datetime64 and
    ## the dbd backend passes epoch seconds.
    return float(u_src[idx]), float(v_src[idx]), t_src[idx]


def _finalise_row(row, mag_var_deg, u_raw, v_raw):
    """Apply the magnetic-variation rotation and fill the DAC columns."""
    row["mag_var_deg"] = mag_var_deg
    row["u_dac_raw"] = u_raw
    row["v_dac_raw"] = v_raw

    if np.isfinite(u_raw) and np.isfinite(v_raw) and np.isfinite(mag_var_deg):
        ## Reuse the package's existing rotation so both backends and the
        ## original notebook workflow stay in step.
        _, u_cor, v_cor = mag_var_correction(0.0, u_raw, v_raw, mag_var_deg)
        row["u_dac"] = float(u_cor)
        row["v_dac"] = float(v_cor)
    else:
        row["u_dac"] = np.nan
        row["v_dac"] = np.nan

    ## Midpoint position, handy for gridding and quiver plots.
    row["mid_lat"] = _nanmean([row["start_lat"], row["end_lat"]])
    row["mid_lon"] = _nanmean([row["start_lon"], row["end_lon"]])
    return row


def _is_nan(value):
    """True for a bare NaN, False for a datetime64 or any real value."""
    return isinstance(value, float) and not np.isfinite(value)


def _nanmean(values):
    """np.nanmean that returns NaN for an all-NaN input without warning."""
    arr = np.asarray(values, dtype=float)
    good = np.isfinite(arr)
    if not good.any():
        return np.nan
    return float(arr[good].mean())


def _order_segments(df):
    """Return the segment table with the agreed column order, sorted by time."""
    df = df.reindex(columns=SEGMENT_COLUMNS)
    ## Pin the time columns so an all-empty backend still matches the contract.
    for column in ("start_time", "end_time", "dac_time"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["source_file"] = df["source_file"].astype(str)
    df = df.sort_values("start_time").reset_index(drop=True)
    return df


##################################################################################################
# ERDDAP backend
##################################################################################################

def segment_table_from_gdf(gdf, dac_source="last"):
    """
    Build the segment table from a flat glider dataframe.

    This reproduces, for every segment at once, the scalar extraction that the
    original processing notebook performed inline inside its per-segment loop.

    Parameters
    ----------
    gdf : pandas.DataFrame
        Must carry the columns ``time``, ``latitude``, ``longitude``, ``depth``,
        ``source_file``, ``m_water_vx``, ``m_water_vy`` and ``m_gps_mag_var``,
        with ``m_gps_mag_var`` already in degrees.
    dac_source : {'last', 'final', 'first'}
        See :func:`_pick_dac`.  'final' is unavailable here because ERDDAP
        trajectory datasets do not normally serve m_final_water_vx.

    Returns
    -------
    pandas.DataFrame
        One row per ``source_file``, columns :data:`SEGMENT_COLUMNS`.
    """
    if dac_source == "final":
        raise ValueError(
            "dac_source='final' needs m_final_water_vx, which the erddap "
            "backend does not fetch; use the dbd backend or dac_source='last'"
        )

    rows = []
    for source_file, seg in gdf.groupby("source_file", sort=False):
        row = _empty_segment_row(source_file)

        t = pd.to_datetime(seg["time"]).dt.tz_localize(None).values
        row["start_time"] = t[0]
        row["end_time"] = t[-1]
        row["n_records"] = int(np.isfinite(seg["depth"].values).sum())
        row["max_depth"] = float(np.nanmax(seg["depth"].values)) \
            if np.isfinite(seg["depth"].values).any() else np.nan

        u_wv = seg["m_water_vx"].values.astype(float)
        v_wv = seg["m_water_vy"].values.astype(float)
        t_wv = t
        nan = np.array([np.nan])
        u_raw, v_raw, t_dac = _pick_dac(t_wv, u_wv, v_wv, nan, nan, nan, dac_source)
        row["dac_time"] = np.datetime64("NaT") if _is_nan(t_dac) else t_dac

        ## Endpoint positions, taken at the first and last finite DAC record to
        ## match the original notebook.
        good = np.where(np.isfinite(u_wv) & np.isfinite(v_wv))[0]
        if len(good):
            row["start_lat"] = float(seg["latitude"].values[good[0]])
            row["start_lon"] = float(seg["longitude"].values[good[0]])
            row["end_lat"] = float(seg["latitude"].values[good[-1]])
            row["end_lon"] = float(seg["longitude"].values[good[-1]])

        mag_var_deg = float(np.nanmean(seg["m_gps_mag_var"].values))
        rows.append(_finalise_row(row, mag_var_deg, u_raw, v_raw))

    return _order_segments(pd.DataFrame(rows))


def load_glider_erddap(ds_id, server, variables=None, constraints=None,
                       dac_source="last"):
    """
    Load glider data from an ERDDAP tabledap dataset.

    Wraps :func:`slocum_ad2cp.get_erddap_dataset` and applies the unit
    conversions the original workflow applied by hand: magnetic variation and
    heading arrive in radians and are converted to degrees, and heading is
    corrected for the deployment-mean declination.

    Parameters
    ----------
    ds_id : str
        ERDDAP dataset ID, e.g. 'ru29-20240419T1430-trajectory-raw-delayed'.
    server : str
        ERDDAP base URL.
    variables : list of str, optional
        Defaults to :data:`ERDDAP_VARIABLES`.
    constraints : dict, optional
        Passed through to erddapy.
    dac_source : str
        See :func:`_pick_dac`.

    Returns
    -------
    (gdf, segments) : (pandas.DataFrame, pandas.DataFrame)
        The flat record frame and the segment table.
    """
    from .analysis import get_erddap_dataset

    variables = list(variables or ERDDAP_VARIABLES)
    gdf = get_erddap_dataset(ds_id, server=server, variables=variables,
                             constraints=constraints, filetype="dataframe")
    ## erddapy appends unit suffixes such as "depth (m)"; restore plain names.
    gdf.columns = variables

    gdf["time"] = pd.to_datetime(gdf.time)
    gdf["m_gps_mag_var"] = np.rad2deg(gdf["m_gps_mag_var"].values)
    gdf["m_heading"] = np.rad2deg(gdf["m_heading"].values)
    gdf["m_heading"] = gdf.m_heading - np.nanmean(gdf.m_gps_mag_var)

    return gdf, segment_table_from_gdf(gdf, dac_source=dac_source)


##################################################################################################
# dbd backend
##################################################################################################

## Flight extensions in order of preference.  Uncompressed files are preferred
## because they are faster to read; where a segment exists in both forms only
## one is kept, otherwise the segment would be processed twice.
FLIGHT_EXTENSIONS = ["dbd", "dcd", "mbd", "mcd", "sbd", "scd"]
SCIENCE_EXTENSIONS = ["ebd", "ecd", "nbd", "ncd", "tbd", "tcd"]


def list_dbd_segments(flight_dir, science_dir=None,
                      flight_extensions=None, science_extensions=None):
    """
    Enumerate glider segments from a pair of card dumps.

    One Dinkum binary file is one segment, so the file listing *is* the segment
    listing.  Slocum cards routinely hold the same segment in more than one
    form, for example an uncompressed ``.dbd`` alongside its compressed
    ``.dcd``.  Files are keyed on their 8.3 basename so each segment appears
    exactly once, taking the first extension available in preference order.

    Parameters
    ----------
    flight_dir : str
        Directory holding the flight-computer logs.
    science_dir : str, optional
        Directory holding the science-bay logs.  Often a different card dump,
        which is why it is a separate argument rather than being derived.
    flight_extensions, science_extensions : list of str, optional
        Extensions to consider, most preferred first.

    Returns
    -------
    pandas.DataFrame
        Columns ``segment``, ``flight_file`` and ``science_file``, sorted by
        segment name.  ``science_file`` is None where no partner exists.
    """
    flight_extensions = flight_extensions or FLIGHT_EXTENSIONS
    science_extensions = science_extensions or SCIENCE_EXTENSIONS

    def _index(directory, extensions):
        found = {}
        if directory is None:
            return found
        for ext in extensions:
            for pattern in (ext, ext.upper()):
                for path in glob.glob(os.path.join(directory, "*." + pattern)):
                    key = os.path.splitext(os.path.basename(path))[0]
                    ## First extension in preference order wins.
                    found.setdefault(key, path)
        return found

    flight = _index(flight_dir, flight_extensions)
    science = _index(science_dir, science_extensions)

    rows = [{"segment": key,
             "flight_file": flight[key],
             "science_file": science.get(key)}
            for key in sorted(flight)]
    return pd.DataFrame(rows, columns=["segment", "flight_file", "science_file"])


def _read_flight_segment(path, cache_dir, params):
    """Read one flight file, returning {param: (time, value)} plus metadata."""
    import dbdreader

    dbd = dbdreader.DBD(path, cacheDir=cache_dir)
    try:
        available = set(dbd.parameterNames)
        wanted = [p for p in params if p in available]
        data = {}
        if wanted:
            result = dbd.get(*wanted, decimalLatLon=True, discardBadLatLon=True)
            ## dbdreader returns a bare tuple for a single parameter.
            if len(wanted) == 1:
                result = [result]
            data = dict(zip(wanted, result))
        empty = (np.array([]), np.array([]))
        for p in params:
            data.setdefault(p, empty)
        meta = {"mission": dbd.get_mission_name(),
                "fileopen": dbd.get_fileopen_time()}
    finally:
        dbd.close()
    return data, meta


def load_glider_dbd(flight_dir, science_dir=None, flight_cache=None,
                    science_cache=None, dac_source="last", segments=None,
                    return_gdf=True, verbose=True):
    """
    Load glider data straight from the raw Dinkum binary files.

    Flight and science logs normally live on separate card dumps with separate
    dbdreader caches, so they are opened independently rather than relying on
    dbdreader's file complementing, which resolves a partner file only within
    the same directory.

    Parameters
    ----------
    flight_dir : str
        Directory of flight-computer logs (dbd/dcd/mbd/mcd/sbd/scd).
    science_dir : str, optional
        Directory of science-bay logs.  Not needed for the segment table; used
        by :func:`load_glider_science_dbd`.
    flight_cache, science_cache : str, optional
        dbdreader cache directories holding the matching ``.cac`` files.
    dac_source : {'last', 'final', 'first'}
        Which logged depth-averaged current represents the segment.  See
        :func:`_pick_dac`.
    segments : pandas.DataFrame, optional
        Output of :func:`list_dbd_segments`.  Built automatically if omitted.
    return_gdf : bool
        Also return a per-record frame of time, position and depth.  Useful for
        track plots and profile gridding; set False to save memory.
    verbose : bool
        Print progress every 100 segments.

    Returns
    -------
    (gdf, segments) : (pandas.DataFrame or None, pandas.DataFrame)
        The per-record frame (or None) and the segment table.
    """
    if segments is None:
        segments = list_dbd_segments(flight_dir, science_dir)
    if len(segments) == 0:
        raise ValueError("no glider segments found in {}".format(flight_dir))

    rows = []
    gdf_parts = []

    for count, seg in enumerate(segments.itertuples(index=False)):
        if verbose and count % 100 == 0:
            print("  reading segment {} of {}".format(count, len(segments)))

        row = _empty_segment_row(seg.segment)
        try:
            data, meta = _read_flight_segment(seg.flight_file, flight_cache,
                                              FLIGHT_PARAMS)
        except Exception as exc:                      # noqa: BLE001
            print("  skipping {}: {}".format(seg.segment, exc))
            rows.append(_finalise_row(row, np.nan, np.nan, np.nan))
            continue

        t_dep, depth = data["m_depth"]
        if len(t_dep) == 0:
            rows.append(_finalise_row(row, np.nan, np.nan, np.nan))
            continue

        row["start_time"] = _epoch_to_datetime64(t_dep[0])
        row["end_time"] = _epoch_to_datetime64(t_dep[-1])
        row["n_records"] = int(len(t_dep))
        row["max_depth"] = float(np.nanmax(depth)) if len(depth) else np.nan

        ## Depth-averaged current
        t_wv, u_wv = data["m_water_vx"]
        _, v_wv = data["m_water_vy"]
        t_fin, u_fin = data["m_final_water_vx"]
        _, v_fin = data["m_final_water_vy"]
        u_wv, v_wv = _align_pair(u_wv, v_wv)
        u_fin, v_fin = _align_pair(u_fin, v_fin)

        u_raw, v_raw, t_dac = _pick_dac(t_wv, u_wv, v_wv, t_fin, u_fin, v_fin,
                                        dac_source)
        row["dac_time"] = _epoch_to_datetime64(t_dac)
        row["u_dac_final"] = float(u_fin[-1]) if len(u_fin) else np.nan
        row["v_dac_final"] = float(v_fin[-1]) if len(v_fin) else np.nan

        ## Magnetic variation is logged in radians and only changes at the
        ## surface, so the segment mean is the right summary.
        _, mag_var = data["m_gps_mag_var"]
        mag_var_deg = np.rad2deg(_nanmean(mag_var)) if len(mag_var) else np.nan

        ## Endpoint positions from the GPS fixes bracketing the segment,
        ## falling back to dead-reckoned position where no fix was acquired.
        t_gps, lat_gps = data["m_gps_lat"]
        _, lon_gps = data["m_gps_lon"]
        lat_gps, lon_gps = _align_pair(lat_gps, lon_gps)
        good = np.isfinite(lat_gps) & np.isfinite(lon_gps)
        if good.any():
            idx = np.where(good)[0]
            row["start_lat"] = float(lat_gps[idx[0]])
            row["start_lon"] = float(lon_gps[idx[0]])
            row["end_lat"] = float(lat_gps[idx[-1]])
            row["end_lon"] = float(lon_gps[idx[-1]])

        rows.append(_finalise_row(row, mag_var_deg, u_raw, v_raw))

        if return_gdf:
            gdf_parts.append(_segment_gdf(seg.segment, t_dep, depth, data,
                                          mag_var_deg))

    table = _order_segments(pd.DataFrame(rows))

    gdf = None
    if return_gdf and gdf_parts:
        gdf = pd.concat(gdf_parts, ignore_index=True).sort_values("time")
        gdf = gdf.reset_index(drop=True)

    return gdf, table


def _align_pair(a, b):
    """Trim two parameter arrays to a common length, NaN-padding if empty."""
    n = min(len(a), len(b))
    if n == 0:
        return np.array([]), np.array([])
    return np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)


def _epoch_to_datetime64(t):
    """Seconds since the epoch to a tz-naive numpy datetime64, NaN safe."""
    if t is None or _is_nan(t):
        return np.datetime64("NaT")
    return np.datetime64(int(round(float(t) * 1e6)), "us")


def _segment_gdf(segment, t_dep, depth, data, mag_var_deg=np.nan):
    """
    Per-record frame for one segment on the m_depth time base.

    Heading, pitch and roll are logged in radians by the flight computer and
    are converted to degrees here so that this frame matches the one the erddap
    backend produces.  Heading is additionally corrected for the segment's
    magnetic variation, as the original workflow did.
    """
    frame = pd.DataFrame({
        "time": _epoch_to_datetime64_array(t_dep),
        "depth": np.asarray(depth, dtype=float),
        "source_file": segment,
    })
    ## Dead-reckoned position, interpolated onto the depth time base.  m_lat and
    ## m_lon update far more slowly than m_depth, so interpolation rather than
    ## a join is the right treatment.
    for out, key in (("latitude", "m_lat"), ("longitude", "m_lon")):
        t_p, v_p = data[key]
        frame[out] = _interp_param(t_dep, t_p, v_p)
    for key in ("m_heading", "m_pitch", "m_roll"):
        t_p, v_p = data[key]
        frame[key] = np.rad2deg(_interp_param(t_dep, t_p, v_p))
    if np.isfinite(mag_var_deg):
        frame["m_heading"] = frame["m_heading"] - mag_var_deg
    return frame


def _epoch_to_datetime64_array(t):
    """Seconds since the epoch to tz-naive datetime64, NaN becoming NaT."""
    micros = np.asarray(t, dtype=float) * 1e6
    out = np.full(micros.shape, np.datetime64("NaT"), dtype="datetime64[us]")
    good = np.isfinite(micros)
    out[good] = micros[good].astype("int64").astype("datetime64[us]")
    return out


def _interp_param(t_target, t_src, v_src):
    """Linear interpolation onto t_target, NaN where the source is empty."""
    t_src = np.asarray(t_src, dtype=float)
    v_src = np.asarray(v_src, dtype=float)
    good = np.isfinite(t_src) & np.isfinite(v_src)
    if good.sum() < 2:
        out = np.full(len(t_target), np.nan)
        if good.sum() == 1:
            out[:] = v_src[good][0]
        return out
    return np.interp(t_target, t_src[good], v_src[good],
                     left=np.nan, right=np.nan)


##################################################################################################
# Science (CTD) product
##################################################################################################

## Reading very many Dinkum files through a single MultiDBD has been observed to
## abort inside the C extension somewhere beyond roughly 700 files, so reads are
## chunked.  This also bounds peak memory on long deployments.
MAX_FILES_PER_READ = 300


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _read_ctd_chunked(files, cache_dir, chunk_size=MAX_FILES_PER_READ,
                      verbose=True):
    """Run get_CTD_sync over the files in batches and concatenate the results."""
    import dbdreader

    parts = []
    for n, batch in enumerate(_chunks(files, chunk_size)):
        if verbose:
            print("    CTD batch {} ({} files)".format(n + 1, len(batch)))
        sci = dbdreader.MultiDBD(filenames=batch, cacheDir=cache_dir)
        try:
            ## get_CTD_sync returns (time, conductivity, temperature, pressure).
            parts.append(np.vstack(sci.get_CTD_sync()))
        finally:
            sci.close()
    if not parts:
        raise ValueError("no CTD data read")
    return np.hstack(parts)


def _read_flight_chunked(files, cache_dir, params, chunk_size=MAX_FILES_PER_READ,
                         verbose=True):
    """Read parameters from many flight files in batches.

    Returns {param: (time, value)} with each batch's samples concatenated in
    file order.
    """
    import dbdreader

    collected = {p: ([], []) for p in params}
    for n, batch in enumerate(_chunks(files, chunk_size)):
        if verbose:
            print("    flight batch {} ({} files)".format(n + 1, len(batch)))
        fli = dbdreader.MultiDBD(filenames=batch, cacheDir=cache_dir)
        try:
            available = [p for p in params if fli.has_parameter(p)]
            if not available:
                continue
            result = fli.get(*available, decimalLatLon=True,
                             discardBadLatLon=True)
            if len(available) == 1:
                result = [result]
            for name, (t, v) in zip(available, result):
                collected[name][0].append(np.asarray(t, dtype=float))
                collected[name][1].append(np.asarray(v, dtype=float))
        finally:
            fli.close()

    out = {}
    for name, (ts, vs) in collected.items():
        if ts:
            out[name] = (np.concatenate(ts), np.concatenate(vs))
        else:
            out[name] = (np.array([]), np.array([]))
    return out


def load_glider_science_dbd(science_dir, flight_dir=None, science_cache=None,
                            flight_cache=None, segments=None, mean_lat=None,
                            verbose=True):
    """
    Build a CTD dataset from the science card, positioned from the flight card.

    Slocum science computers log pressure in bar; it is converted to dbar here.
    Derived quantities are computed with TEOS-10 via ``gsw``.

    Parameters
    ----------
    science_dir : str
        Directory of science-bay logs (ebd/ecd/nbd/ncd/tbd/tcd).
    flight_dir : str, optional
        Directory of flight logs, used to supply position and glider depth.
    science_cache, flight_cache : str, optional
        dbdreader cache directories.
    segments : pandas.DataFrame, optional
        Output of :func:`list_dbd_segments`; rebuilt if omitted.
    mean_lat : float, optional
        Fallback latitude for the pressure-to-depth conversion where the flight
        card gives no position.
    verbose : bool
        Print progress.

    Returns
    -------
    xarray.Dataset
        Dimension ``time``, with conductivity, temperature, pressure, depth,
        practical and absolute salinity, conservative temperature, sigma0, and
        position.
    """
    import gsw
    import xarray as xr

    if segments is None:
        segments = list_dbd_segments(flight_dir, science_dir)

    science_files = [f for f in segments["science_file"].tolist() if f]
    if not science_files:
        raise ValueError("no science files found in {}".format(science_dir))

    if verbose:
        print("  reading {} science files".format(len(science_files)))

    t, cond, temp, pres = _read_ctd_chunked(science_files, science_cache,
                                            verbose=verbose)

    ## Batches are concatenated in file order; sort so time is monotonic.
    order = np.argsort(t)
    t, cond, temp, pres = t[order], cond[order], temp[order], pres[order]

    ## Slocum logs sci_water_pressure in bar.
    pres = pres * 10.0

    ## Position from the flight card, interpolated onto the CTD clock.
    lat = np.full(len(t), np.nan)
    lon = np.full(len(t), np.nan)
    glider_depth = np.full(len(t), np.nan)
    if flight_dir is not None:
        flight_files = segments["flight_file"].tolist()
        if verbose:
            print("  reading {} flight files for position".format(len(flight_files)))
        nav = _read_flight_chunked(flight_files, flight_cache,
                                   ["m_lat", "m_lon", "m_depth"],
                                   verbose=verbose)
        lat = _interp_param(t, *nav["m_lat"])
        lon = _interp_param(t, *nav["m_lon"])
        glider_depth = _interp_param(t, *nav["m_depth"])

    lat_for_gsw = np.where(np.isfinite(lat), lat,
                           mean_lat if mean_lat is not None else 0.0)
    lon_for_gsw = np.where(np.isfinite(lon), lon, 0.0)

    ## Conductivity is logged in S/m; gsw wants the ratio to the standard.
    SP = gsw.SP_from_C(cond * 10.0, temp, pres)
    SA = gsw.SA_from_SP(SP, pres, lon_for_gsw, lat_for_gsw)
    CT = gsw.CT_from_t(SA, temp, pres)
    sigma0 = gsw.sigma0(SA, CT)
    depth = -gsw.z_from_p(pres, lat_for_gsw)

    ds = xr.Dataset(
        data_vars=dict(
            conductivity=("time", cond),
            temperature=("time", temp),
            pressure=("time", pres),
            depth=("time", depth),
            glider_depth=("time", glider_depth),
            salinity=("time", SP),
            absolute_salinity=("time", SA),
            conservative_temperature=("time", CT),
            sigma0=("time", sigma0),
            latitude=("time", lat),
            longitude=("time", lon),
        ),
        coords=dict(time=_epoch_to_datetime64_array(t)),
    )

    ds["conductivity"].attrs = dict(units="S m-1", long_name="sea water electrical conductivity")
    ds["temperature"].attrs = dict(units="degree_Celsius", long_name="sea water temperature")
    ds["pressure"].attrs = dict(units="dbar", long_name="sea water pressure")
    ds["depth"].attrs = dict(units="m", long_name="depth of CTD sample", positive="down")
    ds["glider_depth"].attrs = dict(units="m", long_name="glider depth from m_depth", positive="down")
    ds["salinity"].attrs = dict(units="1", long_name="sea water practical salinity")
    ds["absolute_salinity"].attrs = dict(units="g kg-1", long_name="absolute salinity (TEOS-10)")
    ds["conservative_temperature"].attrs = dict(units="degree_Celsius", long_name="conservative temperature (TEOS-10)")
    ds["sigma0"].attrs = dict(units="kg m-3", long_name="potential density anomaly referenced to 0 dbar")
    ds["latitude"].attrs = dict(units="degrees_north")
    ds["longitude"].attrs = dict(units="degrees_east")

    return ds.sortby("time")


##################################################################################################
# Dispatcher
##################################################################################################

def load_glider(source="dbd", **kwargs):
    """
    Load glider data from the requested backend.

    Parameters
    ----------
    source : {'dbd', 'erddap'}
        Which backend to use.
    **kwargs
        Passed through to :func:`load_glider_dbd` or
        :func:`load_glider_erddap`.

    Returns
    -------
    (gdf, segments) : (pandas.DataFrame, pandas.DataFrame)
    """
    if source == "dbd":
        return load_glider_dbd(**kwargs)
    if source == "erddap":
        return load_glider_erddap(**kwargs)
    raise ValueError("source must be 'dbd' or 'erddap', got {!r}".format(source))


__all__ = [
    "SEGMENT_COLUMNS",
    "FLIGHT_PARAMS",
    "ERDDAP_VARIABLES",
    "FLIGHT_EXTENSIONS",
    "SCIENCE_EXTENSIONS",
    "segment_table_from_gdf",
    "load_glider_erddap",
    "list_dbd_segments",
    "load_glider_dbd",
    "load_glider_science_dbd",
    "load_glider",
    "MAX_FILES_PER_READ",
]
