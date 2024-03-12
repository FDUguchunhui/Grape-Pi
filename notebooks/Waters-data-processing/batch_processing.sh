#!/bin/bash

# Define the regular expression
regex='^.*[0-9]+\.csv$'

# Loop through all files in the current directory
for file in *; do
  # Check if the file matches the regular expression
  if [[ $file =~ $regex ]]; then
    # If the file matches
    echo "$file"
    python /Users/cgu3/Documents/Grape-Pi/analysis-code/Waters-data-processing/calc_local_fdr.py $file -f /Users/cgu3/Documents/Grape-Pi/data/miscellaneous/uniprotkb_swiss_prot.tsv
  fi
done