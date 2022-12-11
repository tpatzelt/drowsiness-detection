## CAN NEURAL NETWORKS BEAT ENGINEERED FEATURES AT DETECTING DROWSY STUDENTS?

This is the code repository accompanying my Individiual Research Project which is part of the
Cognitive Systems Master at University of Potsdam.

I will be using a dataset comprised of eye tracking and closure signal of students in a baseline or
sleep
deprived setting. Using the eye closure signal I want to compare the quality of engineered features
with the quality of features learnt by neural networks during training. The quality of the features
is quantified by their power to predict the drowsiness state of subjects.

In the file `./run.sh` an example call to `./run_grid_search_experiments.py` is shown. The file
contains the code to run the experiments along with the configuration for each experiment.   
The results are saved to `./logs/`. The logs for finished runs from which some made it to paper, are
already saved in `./logs_to_keep`.

In order to run experiments, the original data needs to be preprocessed. This can be done in the
notebook `./notebooks/create_preprocessed_data.ipynb`. In the first cell, the path to the data can
be set and the preprocessed data should be saved under `./data/preprocessed/`.

For inspecting the results please take a look at the
notebook `./notebooks/results/analysis/create_roc_curve_from_predictions.ipynb`.  
