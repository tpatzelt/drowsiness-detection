We recorded the eyes of participants during an eye tracking experiment with an infrared camera at a sampling frequency of 30 Hz. 
Each participant was recorded over three experimental sessions: in normal constitution, sleep-deprived and under the influence of alcohol. The order of the different conditions varies for each subject.

In this data set we provide the eye-closure signal extracted from each video frame and supplementary files with information about:
alcohol consumption, breath alcohol level, subjective level of sleepiness, and the result of a reaction time test.  

- File name format: subjectId_session_sessionType.json

session: 	N-th recording session of this subject (1-3)

sessionType: 	Participant was recorded when being sleep deprived (s), under influence of alcohol (a), in normal constitution (b)


- Each file is a dictionary consisting of information about each video frame in this recording session:

index:		Index of the video frame within this recording. 

eye_closure:	A numeric value in the interval [0,1] for the left, right and both eyes combined. 
		0: eye is open, 1: eye is closed

eye_state:	Integer in {0,1,2,3,4,5} for the left, right and both eyes combined. 
		0 Open
		1 Close
		2 Partially open
		3 Downcast
		4 Not visible
		5 Unknown

- Additionally, we provide the following information for each file:
	1. 	The time (given by index of start and end frame) of alcohol consumptions in ml; and breath measurements with corresponding result. 
		(see files subjectId_session_sessionType_alcohol_consumptions.csv, subjectId_session_sessionType_alcohol_measurements.csv)

	2. 	The time (given by index of start and end frame) of the assessment of the subjective level of sleepiness by Karolinska sleepiness scale;
		response of subject is an integer 1-9, where 1 is "extremely alert" and 9 is "very sleepy"
		(see file subjectId_session_sessionType_karolinska.csv)

	3. 	The time (given by index of start and end frame) and result of a PVT reaction time test.
		(see files subjectId_session_sessionType_pvt_reaction_times.csv, subjectId_session_sessionType_pvt_scores.csv)
