# Instructions to download the voter data:

goto [this website](https://voteview.com/data) and for each `Data Type` in the dropdown list select `Chamber`= Both, `Congress`=All, `File Format`=CSV.

then putt the 4 files into the folder `TDA_practical/data/.`.

Finally in the file `HSall_votes.csv` (attention takes a lot of ram to open ~ 7GB) search for the string `N/AN/AN/AN/A...` and replace it's occurances just by `NA` otherwise this collumn can't be loaded correctly with the script `TDA_practical/exploratory/data_csv_structure.ipynb`.

