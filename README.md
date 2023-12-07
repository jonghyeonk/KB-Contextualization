# Descriptions

### Approximating Multi-Perspective Trace Alignment Using Trace Encodings (Accepted in BPM 2023)

This repository contains the scripts developed for the proposed approximate approach of multi-perspective alignment.

We evaluated the approach by using Sepsis and Road Fines event logs, and the results in Table 3~6 can be obtained by implementing below two scripts.

1. 'prepare_bs_encoding.ipynb' 

    This script shows the processes (i) to generate non-complying traces by modifying the original traces, (ii) to save ground truth of original traces before modification, and (iii) to encode the prepared experimental datasets.

2. Folder 'non_schatistic' includes datasets in non-stochastic setting and experimental codes

    This script shows (i) the approximate approach of multi-perspective alignment and (ii) the results by implementing it (the summary of the results is seen in table 3~6 in our paper).
