# Malezieux_CellRep_2020

Analysis codes for Malezieux, Kees, and Mulle (2020). Cell Reports

### Raw data files (.abf) from the primary population of cells were processed using the following scripts:

* Malezieux_CellRep_load_data.py

   *performs brain state classification, downsamples and smooths membrane potential and LFP traces*

* Malezieux_CellRep_load_data_CS.py

   *detects spikes and complex spikes*

* Malezieux_CellRep_load_data_hf.py

   *analyzes theta coherence between membrane potential and LFP traces*

### Next, these datasets were combined and further processed using the following script:

* Malezieux_CellRep_combine_datasets.py

*The python dictionaries that result from the above script can be found for public download at [insert link here]*

### Other raw data files (.abf) from a separate population of cells subjected to the Rin protocol were processed using the following script:

* Malezieux_CellRep_load_data_Rin.py

*The python dictionaries that result from the above script can be found for public download at [insert link here]*

### The remaining scripts can be run on these python dictionaries to create the main data figures and compute the statistics found in the paper.

*Note: the .yaml file is provided to facilitate the duplication of the Python environment used to write the code*

### Metadata is found in the .xlxs file

* sheet 'Recorded cells' has the list of cells in the main dataset and their properties (see Tables S1 and S2 from the paper)

* sheet 'Dataset keys' is a list and explanation of the keys of the dictionaries provided for each cell in the main dataset

* sheet 'Dataset_Rin keys' is a list and explanation of the keys of the dictionaries provided for each cell in the input resistance dataset

