Name: Arnab Dey
Student ID: 5563169
email: dey00011@umn.edu

#################################################################
# INSTRUCTION TO RUN CODE
#################################################################
Q2: Run 'script_2.py' to run the code for Q2
Q3: Run 'script_3.py' to run the code for Q3

#################################################################
# ADDITIONAL DETAILS
#################################################################
1. Implementation of the Tensorflow model is in 'model.py'.
2. Implementation of Tensorflow callback is in 'utils.py'. This callback handles the convergence criteria and convergence time calculation.
3. I have tested my code on both CPU and GPU on Google Colab. Both worked and produced similar result.
4. Console output for Q2 is in 'q2_console_log.txt'
5. Console output for Q3 with single run in part (b) is in 'q3_console_log.txt'
6. Console output for Q3 with 10 runs in part (b) to calculate the average convergence time is in 'q3_console_log_10_runs.txt'

#################################################################
# PLOT INSTRUCTIONS
#################################################################
1. In 'cnn.py' and 'neural_net.py', there are two control variables for plotting which are by default set to True: 'isPlotReqd' and 'isPlotPdf'.
	If 'isPlotPdf' is True, it creates a folder named 'generatedPlots' in the current directory and saves the plots as pdf there. NOTE that I am
	using Latex styles for the legends on the figure. Therefore, on some machines it might create issues if Latex is not installed properly to
	go along with matplotlib. If it throws error, please set this variable to False, in which case, it will show the plots runtime.
	
	If 'isPlotReqd' is set to False, the codes will not generate any plots. It is used for convenience where only console logs are required.

THE HOMEWORK HAS BEEN DONE COMPLETELY ON MY OWN.