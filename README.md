# Slocum-AD2CP

Slocum-AD2CP is a package designed to process Nortek AD2CP data from Teledyne Webb Research Slocum gliders. The package is designed to walk through the steps to process AD2CP data in several Jupyter notebooks.

Please cite this package using this [DOI](https://doi.org/10.5281/zenodo.7416126). The intention for publishing this code on GitHub is twofold: 1) to make glider based acoustic current profiler data easier to work with and 2) to improve upon the processing workflow and make it more transparent. For these reasons, I encourage comments, questions, concerns, and for users of this code to report any potential bugs. Please do not hesitate to reach out to me at jgradone@marine.rutgers.edu.

<img width="680" alt="Screen Shot 2022-11-30 at 1 01 27 PM" src="https://user-images.githubusercontent.com/43152605/204873998-595184d4-4221-49bf-9134-cc85f56b9bb0.png">

Installation
----------------------
slocum_ad2cp can be install via: <br/>
`pip install slocum_ad2cp`

Processing Raw AD2CP Data
----------------------
This package is built under the assumption that users are processing their AD2CP data to NetCDFs using the Nortek MIDAS software.


Changelog
----------------------
**Unreleased, 2026-09-16**

- Added a `glider.py` module with an interchangeable glider data backend: alongside the existing `get_erddap_dataset` path, `load_glider_dbd`/`load_glider`(`source="dbd"`) reads raw Slocum dbd/ebd files (and their LZ4-compressed dcd/ecd counterparts, needs `dbdreader>=0.6`) directly, useful before a deployment is published to ERDDAP. Both backends return the same segment-table schema.
- Added `correct_ad2cp_mounting` / `detect_roll_offset` to `make_dataset.py`. Some Slocum payload-bay AD2CP mounts report roll 180 degrees from the glider's own frame; left uncorrected, `cell_vert` returns negative cell depths and `binmap_adcp` silently drops every bin. Opt-in, not called by the existing pipeline.
- **Fixed a bug in `beam2enu`** (`make_dataset.py`): the function computed a pitch-dependent 3-of-4-beam selection (dropping the aft beam on downcasts, the forward beam on upcasts, per the glider's actual attitude) and then unconditionally discarded that choice, always applying the beam 2/3/4 transform regardless of pitch. `beam2enu` now takes `honour_pitch_selection` (default `True`, the corrected behaviour).
