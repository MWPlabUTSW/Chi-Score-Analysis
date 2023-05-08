# Chi-Score-Analysis
Code required to run chi-score comparisons of amino acid sequences and intra-sequential analyses of compositional modularity. 

# Instructions
The file containing the complete analysis has three executable cells of Python code:
1. A cell which imports dependencies and defines required functions.
2. A cell in which the seqeunce (and its name) are input and the analysis is performed.
3. A cell in which the desired confidence level is selected and the results are output. 

Cell 1 only needs to be executed once per session. During the analysis in cell 2, all calculated data is stored so that it can be recalled as desired; re-executing cell 2 will erase all stored results from the previous analysis.

Visualization (cell 3) is done by selecting the lowest confidence level you wish to allow in the final solution. For example, if the user inputs a confidence level of 95%, the first solution in which all boundaries are greater than the corresponding z-score (1.96 here) will be selected for the output. If the specified confidence level cannot be achieved, the solution with the greatest mean z-score is output instead.

The data saved during the analysis in cell 2 can be recalled as desired (while stored). Some of the useful items that can be accessed are listed and described below:
- corr_scores: a list of nine pairwise matrices of Pearson's correlation coefficients corresponding to the nine window sizes used in the analysis. By default, the fourth matrix in this list (window size = 11) is used to generate the heatmap in the final output.
- ins_scores: a list containing the insulation scores for each matrix in corr_scores (one for each window size).
- solutions: a list of nine sets of boundaries as defined from each of the nine window sizes
- boundary_groups: the initial set of potential boundary placements as derived from the solutions
- initial_solution: the solution obtained from optimizing the boundaries found in boundary_groups
- trimmed_solution: the solution obtained from optimizing the boundaries after nearby groups have been merged
- trimmed_groups: the set of potential boundary placements after nearby groups have been merged
- trimmed_zscores: the set of zscores corresponding to the trimmed_solution
- final_solutions: a list that contains all optimized boundary placements following each iterative removal (starting with the first solution with all z-scores > 0)
- final_zscores: a list containing the sets of z-scores corresponding to the solutions in final_solutions
