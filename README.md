## Repository for "Elucidating Linear Programs by Neural Encodings"

## How to generate the data

Run ``run_experiment_model.py`` and ``run_experiment_figures.py`` to create the figures for all but the large-scale LP.

For the large-scale LP, [FRaGenLP](https://github.com/leonid-sokolinsky/BSF-LPP-Generator) was used to create the LP found in the folder "lps" (10000_30.txt).

Then, generate data for the large-scale LP using ``linear_programs_plus.py``.

Then, train the model with ``large_experiment.py``.

For the evaluation, the shell script ``large_eval.sh`` can be run.

## How the large-scale LP was generated

From the repository, the code from the No PMI folder was run.
The code was left as it is, only some parameters in ``Problem-Parameters.h`` were changed to generate a large LP: 
PP_N 10000
PP_NUM_OF_RND_INEQUALITIES 30
PP_MAX_N 10000
PP_ALPHA 100

