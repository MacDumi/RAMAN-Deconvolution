Deconvolution of Raman spectra

Initially written to deconvolute soot spectra

Usage:

	$./Deconvolution.py <FileName> #Displays the result without saving it
	$./Deconvolution.py --save <FileName> #Displays and saves the result
	$./Deconvolution.py -s <FileName> #Displays and saves the result     
	$./Deconvolution.py -p <DirectoryName> #Processes all text files from the directory
	$./Deconvolution.py --path <DirectoryName> #Processes all text files from the directory
	$./Deconvolution.py -f <FileName> #Uses the spike detection routine
	$./Deconvolution.py --filter <FileName> #Uses the spike detection routine

Help:

	$./Deconvolution.py --help
	$./Deconvolution.py -h  

Spike detection (-f / --filter)

The script has an integrated "spike" detector. To detect "spikes" the moving average of two ajacent poins is compared and if the difference is higher than the threshold value, the point is considered as a spike. If the algorithm fails to detect spikes, the user can specify another threshold value and try again.

Polynomial degree for the baseline

By default the baseline will be calculated using a first degree polynomial function. However, the user is able to change the degree of the poly-function if the baseline does not fit the experimental data. A plot with the calculated baseline and the experimental data is displayed after each attempt.
