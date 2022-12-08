# Slocum-AD2CP

Slocum-AD2CP is a package designed to process Nortek AD2CP data from Teledyne Webb Research Slocum gliders. The package is designed to walk through the steps to process AD2CP data in several Jupyter notebooks.

Please cite this package using this DOI. The intention for publishing this code on GitHub is twofold: 1) to make glider based acoustic current profiler data easier to work with and 2) to improve upon the processing workflow and make it more transparent. For these reasons, I encourage comments, questions, concerns, and for users of this code to report any potential bugs. Please do not hesitate to reach out to me at jgradone@marine.rutgers.edu.

<img width="680" alt="Screen Shot 2022-11-30 at 1 01 27 PM" src="https://user-images.githubusercontent.com/43152605/204873998-595184d4-4221-49bf-9134-cc85f56b9bb0.png">


Processing Raw AD2CP Data
----------------------
This package is built under the assumption that users are processing their AD2CP data to NetCDFs using the Nortek MIDAS software.

Work-Flow
----------------------
This package is designed to step the user through the steps to process AD2CP data in several Jupyter notebooks. The notebooks are labeled with the prefix 01_, 02_, 03_, etc. to show the recommended order of operations. There are several examples from some of my on-going research projects.

1) The 01_ notebook is intended for the user to become familiar with their data and some of the processing steps through a quick data exploration.
2) The 02_ notebook is intended for the user to pre-process the AD2CP data. Additional information is included in the notebook but this step reads the AD2CP NetCDFs created by MIDAS, performs a few data manipulation steps, QAQCs the data, and saves the output.
3) The 03_ notebook is intended for the user to take the data processed in the 02_ notebook, read in glider data using ERDDAP, perform a least squares linear inversion to extract horizontal velocity profiles, and save the output. If users do not have their glider data on ERDDAP, this step will require users to read in their data in a different manner and combine it with the existing work flow. Should be plug and play from there.
4) The 04_ notebook is a *bonus* notebook. This step is the analysis! The examples included here are from some of my on-going research projects.


This repository was designed using Cookiecutter Data Science: https://drivendata.github.io/cookiecutter-data-science/
