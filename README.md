# Robustness differences in fine tuning and pre training

In this repository there are all the scripts we have used to perform the tests and record our results. Function definitions can be found in utils.py.

Under each folder there is a `readme.md` that contains additional information on the code. The repository as is cannot be run directly since the pretraining dataset was too large to upload. Furthermore almost every script is supposed to be run from the root directory (where this file is) due to relative paths.

### Images and logs
We have logged both on WandB and with `.json` files. In the repository there is only the latter and therefore they are not complete. Under `Logged robustness` it is possible to find the logs of the fine-tuning and under `Figure` there are many plots we looked at and some of which are also in the report. Other plots can be found in the outputs of `merge_plots.ipynb` together with functions to plot specific logs.
