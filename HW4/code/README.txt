Name: Arnab Dey
Student ID: 5563169
email: dey00011@umn.edu

#################################################################
# INSTRUCTION TO RUN CODE
#################################################################
Q1: Run 'script_1.py' to run the code for Q1
Q2: Run 'script_2.py' to run the code for Q2
Q3: Run 'script_3.py' to run the code for Q3

#################################################################
# ADDITIONAL DETAILS
#################################################################
1. Implementation of the stump is in 'tree.py'.
2. Implementation of data processor is in 'data_processor.py'.
3. Console output for Q1 is in 'q1_console_log.txt'.
4. Console output for Q2 is in 'q2_console_log.txt'.
5. Console output for Q3 is in 'q3_console_log.txt'.
6. For Q3, I iterated over 10 runs for each k to choose the best cluster centers. The best cluster centers are the ones among the 10 runs
	which produced lowest cumulative error.

#################################################################
# PLOT INSTRUCTIONS
#################################################################
1. In 'adaboost.py', 'rf.py' and 'kmeans.py', there are two control variables for plotting which are by default set to True: 'isPlotReqd' and 'isPlotPdf'.
	If 'isPlotPdf' is True, it creates a folder named 'generatedPlots' in the current directory and saves the plots as pdf there. NOTE that I am
	using Latex styles for the legends on the figure. Therefore, on some machines it might create issues if Latex is not installed properly to
	go along with matplotlib. If it throws error, please set isPlotPdf variable to False, in which case, it will show the plots runtime.
	
	If 'isPlotReqd' is set to False, the codes will not generate any plots. It is used for convenience where only console logs are required.

THE HOMEWORK HAS BEEN DONE COMPLETELY ON MY OWN.