source venv/bin/activate
export PYTHONPATH=.
#python drowsiness_detection/run_grid_search_experiment.py with random_forest recording_frequency=30 window_in_sec=60 grid_search_params.n_jobs=-1 num_targets=9;
#python drowsiness_detection/run_grid_search_experiment.py with random_forest recording_frequency=30 window_in_sec=10 grid_search_params.n_jobs=-1 num_targets=2 seed=42;
#python drowsiness_detection/run_grid_search_experiment.py with random_forest recording_frequency=30 window_in_sec=20 grid_search_params.n_jobs=-1 num_targets=2 seed=42 max_depth=60;
#python drowsiness_detection/run_grid_search_experiment.py with lstm recording_frequency=60 window_in_sec=10 num_targets=2 seed=45;
#python drowsiness_detection/run_grid_search_experiment.py with cnn recording_frequency=30 window_in_sec=10 num_targets=2 seed=42;
python drowsiness_detection/run_grid_search_experiment.py with bi-lstm recording_frequency=30 window_in_sec=60 num_targets=2 seed=45;
