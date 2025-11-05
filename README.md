# MMS_codes
Collection of scripts to analyze MMS data.

The src directory contains routines to compute the pressure-strain interaction and related terms (Pi-D, ptheta), as well as routines to compute the scale-filtered terms in the Vlasov-Maxwell equations such as the cross-scale energy fluxes.
'download_demo.py' uses PySPEDAS to download all the required data to compute these quantities and stores them in a .h5 file, which can then be passed as an argument to the computation routines.
