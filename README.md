# drink_smart
Machine Learning project for a beer recommendation engine with Ted Owens

## Scripts

### `profile_data.py`
Usage : `profile_data.py` `[CSV DATA FILES]`

This script expects a series of `.csv` data files and will build dictionaries with frequencies of observations for the beer name, the reviewer and the brewery. Outputs a `.json` representation of the dictionary to be consumed by `merge_json.py` because each file is just a subset of whole data set. (Outputs of this are currently in the `./data/beers`, `./data/breweries` and `./data/reviewers` directories.)

### `merge_json.py`
Usage : `merge_json.py` `[OUTPUT FILE NAME]` `[JSON DICTIONARIES TO MERGE]`

Because putting all of our observations into one dictionary would require an enormous amount of memory and time, we split up the source file into components and then build a dictionary for each subfile. We need to combine all these dictionaries. This file will merge the dictionaries, summing observation counts for each each. (Combined outputs are currently in the `./data/combined_json` directory.)

### `json_to_csv.py`
Usage : `json_to_csv.py` `[JSON FILE TO CONVERT]`

Pretty simple little script to turn the JSON dictionary occurences produced from the `merge_json.py` script into csv files digestible by Matlab.

### profile.m

Reads csv files of beer and reviewer occurences (ie, how many reviews are attributed to a single reviewer or beer respectviely) and plots the cumulative sum. This is handy for understanding where the bulk of the data lies and what threshold for reviews we can responsibly apply when selecting the data we will actually process.
