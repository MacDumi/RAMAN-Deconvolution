Deconvolution of Raman spectra

Initially written to deconvolute soot spectra

Usage:

	$./Deconvolution.py <FileName> #Displays the result without saving it
	$./Deconvolution.py --save <FileName> #Displays and saves the result
	$./Deconvolution.py -s <FileName> #Displays and saves the result     
	$./Deconvolution.py -p <DirectoryName> #Processes all text files from the directory
	$./Deconvolution.py --path <DirectoryName> #Processes all text files from the directory

Help:

	$./Deconvolution.py --help
	$./Deconvolution.py -h  

Spike detection

The script has an integrated "spike" detector. To detect "spikes" the moving average of two ajacent poins is 
compared and if the difference is higher than the threshold value, the point is considered as a spike. If the algorithm 
failes to detect spikes, the user can specify another threshold value and try again.

