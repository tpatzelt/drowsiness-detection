# Alcohol consumptions:
In this block the participant is given alcohol to consume.

file pattern: {subject_id}_{session_id}_{session_type}_alcohol_consumptions.csv:

columns:
- frame_begin: first frame of respective alcohol consumption block
- frame_end: last frame of respective alcohol consumption block


# Alcohol measurements
In this block the blood alcohol content of the participant is estimated from a breath sample.

file pattern: {subject_id}_{session_id}_{session_type}_alcohol_measurements.csv:

columns:
- frame_begin: first frame of respective alcohol measurement block
- frame_end: last frame of respective alcohol measurement block
- promille: measured promille value


# Karolinska sleepiness scale (KSS)
The Karolinska Sleepiness Scale is a subjective scale in which participants indicate their awakeness during the last 10 minutes.
The scale usually ranges from 1 to 10, but ranges from 0 to 9 here for convenience during the experiment.
More info: https://www.med.upenn.edu/cbti/assets/user-content/documents/Karolinska%20Sleepiness%20Scale%20(KSS)%20Chapter.pdf

file pattern: {subject_id}_{session_id}_{session_type}_karolinska.csv:

columns:
- frame_begin: first frame of respective KSS block
- frame_end: last frame of respective KSS block
- response_karolinska: given response in respective KSS block

# PVT reaction times
The Psychomotor vigilance task (PVT) is a sustained-attention, reaction-timed task that measures the speed with which subjects respond to a visual stimulus.
A PVT block takes 10 minutes, with as many trials as the participant manages to do during the experiment block.

file_pattern: {subject_id}_{session_id}_{session_type}_pvt_reaction_times.csv:

columns:
- frame_begin: first frame of respective PVT trial
- frame_end: last frame of respective PVT trial
- reaction_time: reaction time in respective PVT trial
- block_id: corresponding block_id of respective PVT trial


# PVT scores
There is a range of aggregation methods to calculate scores of a list of PVT reaction times.
Each score corresponds to a specific PVT block.

file_pattern: {subject_id}_{session_id}_{session_type}_pvt_reaction_times.csv:

columns:
- frame_begin: first frame of respective PVT block
- frame_end: last frame of respective PVT block
- pvt_n_lapses_500ms: number of reaction-times > 500 ms
- pvt_n_lapses_60s: number of reaction-times > 60 s
- pvt_median_rt: median reaction time
- pvt_mean_rt: mean reaction time
- pvt_mean_log_rt: logarithmic mean reaction time
- pvt_mean_slowest_10_percent_rt: mean of the 10 percent slowest reaction times
- pvt_mean_fastest_10_percent_rt: mean of the 10 percent fastest reaction times


