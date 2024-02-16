# Artificial Neural Reserve

This repository contains code for the manuscript: Too Big to Fail: Larger Language Models are Disproportionately Resilient to Induction of Dementia-Related Linguistic Anomalies, submitted for ACL 2024.

## Setup
Please install dependency packages using  `conda env create -f environment.yml`. The code is developed using Python 3.11 and pytorch 12.1.

While the data of [AD Recognition through Spontaneous Speech (ADReSS)](https://dementia.talkbank.org/ADReSS-2020/)  and [Wisconsin Longitudinal Study (WLS)](https://dementia.talkbank.org/access/English/WLS.html) are publicly available, we are not able to redistribute any of these data per Data Use agreement with Dementia Bank. Individual investigators need to contact the [Dementia Bank](https://dementia.talkbank.org/) to request access to the data.

We use [TRESTLE (Toolkit for Reproducible Execution of Speech Text and Language Experiments)](https://github.com/LinguisticAnomalies/harmonized-toolkit) for the text preprocessing. Please refer to the link for preprocessing details.

Before start, please create a `config.ini` file under the `scripts` folder, using the following template:

```
[PATH]
PrefixManifest = /path/to/transcripts/
wls_text_output = /path/to/wls/text/transcripts/
```

## Folders

The structure of this repo is listed as follows:
```
├── results
├── ft-models
├── scripts
    ├── break_gpt2.py
    ├── eval_wls.py
    ├── util_fun.py
    ├── config.ini
```

- `break_gpt2.py`: the script to a) get the ranking of attention heads in a GPT-2 model using ADReSS training set, and b) evaluate the mask pattern on the ADReSS test set. The ranking of attention heads for each model is saved under `results` folder.
- `eval_wls.py`: the script to estimate perplexity on transcripts produced by healthy individuals from the WLS dataset.
- `util_fun.py`: the scripts containing several helper functions
