# Chi-Score-Analysis
Code required to run chi-score comparisons of amino acid sequences and intra-sequential analyses of compositional modularity. Analysis is available in three formats:
1. A Python module that can be imported directly into an environment running Python3. This file has all functions annotated with their purpose and/or outputs described.
2. A Jupyter Notebook file that walks the user through each step of the analysis, allowing for investigation of the output(s) from each step.
3. A streamlined Jupyter Notebook file that perfoms the entire analysis in one executable cell with user-defined input and parameters.

# Instructions
The Chi-Score Analysis is available as a Python package that give the user free use of the analysis and its many functions. To do this, simply download the chi_score_analysis.py file to an environment running Python3 and all dependencies installed (see below). We recommend importing the module as 'xid', which is used in both available Jupyter Notebook files.

To use the algorithm on a step-by-step basis, the Chi-Score Analysis is available as a multi-cell Jupyter Notebook file that walks the user through and stores the output(s) of each step of the analysis. Each cell is accompanied by a description of what the functions are doing, as well as how to properly use them (i.e.,  formats of the input / output values).

For the most streamlined use of the algorithm, the Chi-Score Analysis is also available as a single-cell Jupyter Notebook file that performs the entire analysis on the input sequence using the following user-defined parameters:
- sequence: the amino acid sequence to be analyzed, input as single string of residues
- window_sizes: the window sizes to use when gathering initial set of boundaries (6, 8, 10, 12, 14, 16, 18, 20, and 22 by default)
- cutoff: the z-score corresponding to the confidence level at which the analysis will stop iteratively removing low-scoring boundaries (1.96, or 95%, by default)
- plot: whether the user would like the results plotted; this will involve additional parameters that the user may adjust which are explained further in the notebook

# Dependencies
The Chi-Score Analysis is dependent on the following Python packages, which must be installed in the environment which the user wishes to run the algorithm:
- jupyter >= 1.0.0
- matplotlib >= 3.7.2
- numpy >= 1.25.2
- pandas >= 2.0.3
- scipy >= 1.11.1
