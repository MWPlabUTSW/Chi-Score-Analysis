
## Manuscript version of Chi-Score Analysis ##


## Import dependencies ##
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import entropy


## Amino acid grouping schemes,named for how many groups residues are distributed among ##
residue_groups = {'two': [['L', 'V', 'I', 'M', 'C', 'A', 'G', 'S', 'T', 'P', 'F', 'Y', 'W'],
                          ['E', 'D', 'N', 'Q', 'K', 'R', 'H']],
                  'three': [['L', 'V', 'I', 'M', 'C', 'A', 'G', 'S', 'T', 'P'],
                            ['F', 'Y', 'W'], ['E', 'D', 'N', 'Q', 'K', 'R', 'H']],
                  'four': [['L', 'V', 'I', 'M', 'C'], ['A', 'G', 'S', 'T', 'P'],
                           ['F', 'Y', 'W'], ['E', 'D', 'N', 'Q', 'K', 'R', 'H']],
                  'five': [['L', 'V', 'I', 'M', 'C'], ['A', 'G', 'S', 'T', 'P'], ['F', 'Y', 'W'],
                           ['E', 'D', 'N', 'Q'], ['K', 'R', 'H']],
                  'six': [['L', 'V', 'I', 'M'], ['A', 'G', 'S', 'T'], ['P', 'H', 'C'],
                          ['F', 'Y', 'W'], ['E', 'D', 'N', 'Q'], ['K', 'R']],
                  'eight': [['L', 'V', 'I', 'M', 'C'], ['A', 'G'], ['S', 'T'], ['P'],
                            ['F', 'Y', 'W'], ['E', 'D', 'N', 'Q'], ['K', 'R'], ['H']],
                  'ten': [['L', 'V', 'I', 'M'], ['C'], ['A'], ['G'], ['S', 'T'], ['P'],
                          ['F', 'Y', 'W'], ['E', 'D', 'N', 'Q'], ['K', 'R'], ['H']],
                  'eleven': [['L', 'V', 'I', 'M'], ['C'], ['A'], ['G'], ['S', 'T'],
                             ['P'], ['F', 'Y', 'W'], ['E', 'D'], ['N', 'Q'], ['K', 'R'], ['H']],
                  'twelve': [['L', 'V', 'I', 'M'], ['C'], ['A'], ['G'], ['S', 'T'],
                             ['P'], ['F', 'Y'], ['W'], [
                                 'E', 'Q'], ['D', 'N'], ['K', 'R'],
                             ['H']],
                  'fifteen': [['L', 'V', 'I', 'M'], ['C'], ['A'], ['G'], ['S'], ['T'], ['P'],
                              ['F', 'Y'], ['W'], ['E'], ['Q'], ['D'], ['N'], ['K', 'R'], ['H']],
                  'eighteen': [['L', 'M'], ['V', 'I'], ['C'], ['A'], ['G'], ['S'], ['T'], ['P'],
                               ['F'], ['Y'], ['W'], ['E'], ['D'], ['N'], ['Q'], ['K'], ['R'], ['H']],
                  'twenty': [['L'], ['M'], ['V'], ['I'], ['C'], ['A'], ['G'], ['S'], ['T'], ['P'],
                             ['F'], ['Y'], ['W'], ['E'], ['D'], ['N'], ['Q'], ['K'], ['R'], ['H']]}


def count_residues_in_sequence(sequence, residue):
    ''' 
    Counts number of 'residue' in 'sequence'
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    residue: single-character string corresponding to amino acid (ex: 'A' for alanine)
    Outputs integer residue count
    '''
    residue_count = 0
    residue_count += sequence.count(residue)
    return residue_count


def get_observed_values(sequence_1, sequence_2, groups='twenty'):
    ''' 
    Computes observed values for chi-score calculation for amino acid groups in compared sequences
    sequence_1/2: string of capitalized amino acid characters (ex: 'MSTAVG...')
    groups: amino acid grouping scheme to use, see 'residue groups' for list of grouping schemes
    Outputs 2 x (# of residue groups) matrix of integer observed values
    '''
    obs = np.zeros((2, len(residue_groups[groups])))
    fracsall1 = list()
    for group in residue_groups[groups]:
        count = 0
        for residue in group:
            count += count_residues_in_sequence(sequence_1, residue)
        fracsall1.append(count)
    obs[0] = fracsall1
    fracsall2 = list()
    for group in residue_groups[groups]:
        count = 0
        for residue in group:
            count += count_residues_in_sequence(sequence_2, residue)
        fracsall2.append(count)
    obs[1] = fracsall2
    return obs


def get_expected_values(observed_values, ratio = 0.5):
    ''' 
    Computes expected values from observed values
    observed_values: 2 x (# of residue groups) matrix of integer observed values
    ratio: length ratio of compared sequences; for intramolecular analyses this should always be 0.5
    Outputs 2 x (# of residue groups) matrix of float expected values
    '''
    exp = np.zeros((2, len(observed_values[0])))
    for seqs in [0, 1]:
        for group in range(0, len(observed_values[0])):
            exp[seqs, group] = (observed_values[0, group] +
                                observed_values[1, group]) * ratio
    return exp


def get_chi_score(observed_values, expected_values, res_conts=False):
    ''' 
    Computes chi-score from observed and expected values
    observed_values: 2 x (# of residue groups) matrix of integer observed values
    expected_values: 2 x (# of residue groups) matrix of float expected values
    res_conts: if TRUE, outputs list of float chi-scores for each residue group
    Outputs chi-score for the input obs. and exp. values
    '''
    chi_scores = np.zeros((2, len(observed_values[0])))
    for seqs in [0, 1]:
        for group in range(len(observed_values[0])):
            if expected_values[seqs, group] != 0:
                chi_scores[seqs, group] = (
                    (observed_values[seqs, group] - expected_values[seqs, group])**2) / expected_values[seqs, group]
            else:
                chi_scores[seqs, group] = 0
    if res_conts is True:
        return [np.sum(chi_scores[:, group]) / np.sum(observed_values) for group in range(len(observed_values[0]))]
    else:
        return np.sum(chi_scores) / np.sum(observed_values)


def calculate_chi_score(sequence_1, sequence_2, groups='twenty'):
    ''' 
    Calculates chi-score between two input sequences
    sequence_1/2: string of capitalized amino acid characters (ex: 'MSTAVG...')
    groups: amino acid grouping scheme to use, see 'residue groups' for list of grouping schemes
    Outputs float chi-score for input sequences
    '''
    obs = get_observed_values(sequence_1, sequence_2, groups)
    exp = get_expected_values(obs)
    return get_chi_score(obs, exp)


def get_heatmap_scores(sequence, window, groups='twenty'):
    '''
    Computes pairwise matrix of subsequence chi-scores for input sequence
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    window: integer value for length of subsequences to use (EVEN INTEGERS RECOMMENDED)
    groups: amino acid grouping scheme to use, see 'residue groups' for list of grouping schemes
    '''
    slen = len(sequence)
    nwin = slen - (window - 1)
    seqs = list()
    for windows in range(0, nwin):
        subseq = sequence[windows:windows + window]
        seqs.append(subseq)
    scores = np.zeros((nwin, nwin))
    for x in range(0, nwin):
        for y in range(0, nwin):
            scores[x, y] = calculate_chi_score(seqs[x], seqs[y], groups)
    return scores


def get_corr_scores(scores):
    ''' 
    Performs Pearson Correlation on pairwise chi-score matrix
    scores: matrix of pairwise subsequence chi-scores
    Outputs matrix of pairwise subsequence correlation coefficients
    '''
    df = pd.DataFrame(scores)
    dfcorr = df.corr()
    corr_scores = dfcorr.to_numpy()
    return corr_scores


def get_insulation_scores(corr_matrix, s):
    ''' 
    Computes insulation scores from matrix of correlation coefficients in square window of size 's'
    corr_matrix: matrix of pairwise subsequence correlation coefficients
    s: side length for sliding square widow
    Outputs list of mean correlation coefficients contained in the window as it slides along the diagonal of corr_matrix
    '''
    n_scores = len(corr_matrix)
    windows = list()
    for x in range(n_scores):
        if x < (s - 1) / 2:
            windows.append(
                corr_matrix[:x + int((s + 1) / 2), :x + int((s + 1) / 2)])
        elif x >= (s - 1) / 2 and x <= n_scores - (s - 2) / 2:
            windows.append(corr_matrix[x - int((s - 1) / 2):x + int(
                (s + 1) / 2), x - int((s - 1) / 2):x + int((s + 1) / 2)])
        else:
            windows.append(
                corr_matrix[x - int((s - 1) / 2):, x - int((s - 1) / 2):])
    return [np.mean(window) for window in windows]


def get_minima(insulation_scores):
    ''' 
    Identifies local minima (potential boundaries) from insulation scores
    insualtion_scores: list of float insualtion scores
    Outputs list of x-values corresponding to significant minima
    '''
    minima = list()
    min_scores = list()
    for x in np.arange(1, len(insulation_scores) - 1, 1):
        if insulation_scores[x] < insulation_scores[x - 1] and insulation_scores[x] < insulation_scores[x + 1] and insulation_scores[x] < np.mean(insulation_scores):
            minima.append(x)
            min_scores.append(insulation_scores[x])
    min_array = np.zeros((len(minima), 2))
    for x in range(len(minima)):
        min_array[x, 0], min_array[x, 1] = minima[x], min_scores[x]
    min_array = min_array[min_array[:, 1].argsort()]
    solution = list()
    blacklist = list()
    for row in min_array:
        if row[0] not in blacklist:
            solution.append(row[0])
            black_range = np.arange(row[0] - 9, row[0] + 10, 1)
            for residue in black_range:
                blacklist.append(residue)
    solution.sort()
    return solution


def subsequence_to_boundary(subsequence_indices, subsequence_length):
    ''' 
    Converts subsequence indices to residue indices (where boundary actually occurs)
    subsequence_indices: the indices of subsequences identified as insulation score minima
    subsequence_length: the length of the subsequences being compared (window used in get_heatmap_scores())
    Outputs indices of potential boundary positions on the original sequence
    '''
    boundary_indices = list()
    for index in subsequence_indices:
        boundary_indices.append(index + (subsequence_length / 2))
    return boundary_indices


def get_region_scores(sequence, optimized_nodes):
    '''
    Computes chi-scores between neighboring regions (separated by a boundary)
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    optimized_nodes: positions of boundaries to be scored
    Outputs list of chi-scores corresponding to input boundaries
    '''
    boundaries = [0]
    for node in optimized_nodes:
        if int(node) not in boundaries:
            boundaries.append(int(node))
    boundaries.append(len(sequence))
    if len(boundaries) == 2:
        return 0
    else:
        nregions = len(boundaries) - 1
        regions = list()
        for region in range(0, nregions):
            regions.append(sequence[boundaries[region]:boundaries[region + 1]])
        scores = list()
        for x in range(0, nregions):
            for y in range(0, nregions):
                if y == x - 1:
                    if len(regions[x]) < 1 or len(regions[y]) < 1:
                        scores.append(0)
                    else:
                        scores.append(calculate_chi_score(
                            regions[x], regions[y]))
        return scores


def get_modules(sequence, boundaries):
    '''
    Returns amino acid sequences of modules in 'sequence' as defined by 'boundaries'
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    boundaries: list of positions denoting where to parse sequence (boundary at 10 splits residues 1-10 from 11-)
    Outputs list of module sequences as strings
    '''
    bounds = [0]
    for bound in boundaries:
        bounds.append(bound)
    bounds.append(len(sequence))
    modules = list()
    for x in np.arange(1, len(bounds), 1):
        modules.append(sequence[int(bounds[x - 1]):int(bounds[x])])
    return modules


def optimize_boundaries(sequence, groups):
    ''' 
    Determines the optimal placement for each boundary in 'groups' for 'sequence'
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    groups: list of lists, each containing numeric boundary positions to consider
    Outputs list of boundary positions, each from a group in 'groups', that maximize intermodular chi-scores
    '''
    start_sols = list()
    for group in groups:
        if group[0] == 0:
            start_sols.append(group[1])
        else:
            start_sols.append(group[0])
    while len(start_sols) < 3:
        start_sols.append(0)
    count = 0
    if len(groups) >= 3:
        end = 2
    elif len(groups) == 2:
        end = 1
    else:
        end = 0
    while count < len(groups) - end:
        xs = groups[count]
        if len(groups) >= 2:
            ys = groups[count + 1]
        else:
            ys = [0]
        if len(groups) >= 3:
            zs = groups[count + 2]
        else:
            zs = [0]
        scores = np.zeros((len(xs), len(ys), len(zs)))
        for x in np.arange(0, len(xs), 1):
            for y in np.arange(0, len(ys), 1):
                for z in np.arange(0, len(zs), 1):
                    start_sols[count], start_sols[count +
                                                  1], start_sols[count + 2] = xs[x], ys[y], zs[z]
                    scores[x, y, z] = np.mean(
                        get_region_scores(sequence, start_sols))
        best_index = np.column_stack(np.where(scores == np.max(scores)))
        start_sols[count], start_sols[count + 1], start_sols[count +
                                                             2] = xs[best_index[0][0]], ys[best_index[0][1]], zs[best_index[0][2]]
        count = count + 1
    optimized_sols = list()
    for boundary in start_sols:
        if boundary != 0:
            optimized_sols.append(boundary)
    return optimized_sols


def get_complexity(sequence, window):
    ''' 
    Computes local sequence complexities as Shannon Entropy in 'sequence' with specified 'window' size
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    window: length of subsequences for which each entropy is calculated
    Outputs list of entropies corresponding to each subsequence
    '''
    nwin = len(sequence) - (window - 1)
    subseqs = list()
    for windows in np.arange(0, nwin, 1):
        subseq = sequence[windows:windows + window]
        subseqs.append(subseq)
    complexities = list()
    for seq in subseqs:
        pk = list()
        for res in ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']:
            pk.append(count_residues_in_sequence(seq, res) / len(seq))
        complexities.append(entropy(pk))
    return complexities


def cluster_boundaries(window_groups):
    '''
    Takes boundaries clustered by window size and clusters them by residue position
    window_groups: the boundary positions as determined by different window sizes, input as list of lists
    Outputs boundary positions clustered spatially, so that potential placements of the same boundary are grouped together
    '''
    boundaries0 = window_groups[0].copy()
    boundary_groups = list()
    for boundary in boundaries0:
        boundary_groups.append([boundary])
    for group in window_groups:
        if group != boundaries0:
            for node in group:
                diffs = list()
                for boundary in boundary_groups:
                    diffs.append(abs(boundary[-1] - node))
                ## If boundary is at least 5 residues from nearest residue in an established group, a new group is defined ##
                if np.min(diffs) >= 5:
                    boundary_groups.append([node])
                else:
                    node_index = np.column_stack(
                        np.where(diffs == np.min(diffs)))
                    boundary_groups[node_index[0][0]].append(node)
    sorted_groups = list()
    for group in boundary_groups:
        new_group = list()
        for node in group:
            new_group.append(node)
        new_group.sort()
        sorted_groups.append(new_group)
    sorted_groups.sort()
    return sorted_groups


def eliminate_short_modules(sequence, initial_solution, boundary_groups, cutoff=6):
    ''' 
    Removes short modules from the solution by merging nearby boundary groups
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    initial_solution: list of boundary positions optimized by optimize_boundaries()
    bounadry_groups: list of groups containing boundary positions clustered by cluster_boundaries()
    cutoff: shortest module to allow in output
    Outputs updated solution (list of re-optimized positions) and boundary groups (some will have been merged into one)
    '''
    module_lengths = list()
    module_lengths.append(initial_solution[0])
    for x in np.arange(1, len(initial_solution), 1):
        module_lengths.append(initial_solution[x] - initial_solution[x - 1])
    shortest_mod = np.min(module_lengths)
    considered_groups = list()
    for group in boundary_groups:
        considered_groups.append(group.copy())
    temporary_solution = initial_solution.copy()
    while shortest_mod < cutoff:
        module_lengths = list()
        module_lengths.append(temporary_solution[0])
        for x in np.arange(1, len(temporary_solution), 1):
            module_lengths.append(
                temporary_solution[x] - temporary_solution[x - 1])
        index = module_lengths.index(shortest_mod)
        new_groups = list()
        for x in np.arange(0, len(considered_groups), 1):
            if x != index and x + 1 != index:
                new_groups.append(considered_groups[x])
            elif x + 1 == index:
                new_group = considered_groups[x].copy()
                for node in considered_groups[x + 1]:
                    new_group.append(node)
                new_groups.append(new_group.copy())
        temporary_solution = optimize_boundaries(sequence, new_groups)
        considered_groups = list()
        for group in new_groups:
            considered_groups.append(group.copy())
        module_lengths = list()
        module_lengths.append(temporary_solution[0])
        for z in np.arange(1, len(temporary_solution), 1):
            module_lengths.append(
                temporary_solution[z] - temporary_solution[z - 1])
        shortest_mod = np.min(module_lengths)
    return temporary_solution.copy(), considered_groups.copy()


def get_zscores(sequence, boundaries, affected_boundaries=[], z_scores=[]):
    ''' 
    Computes the z-scores for each boundary in sequence
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    boundaries: list of boundary positions identified in sequence
    affected_boundaries: list of boundary positions for which to calculate z-scores; used during analysis to speed up the trimming step by not re-scoring boundaries that have not changed
    z_scores: list of initial z-scores; boundaries not in affected_boundaries will be assigned the corresponding z-score from this list
    Outputs list of float z-scores corresponding to the input boundaries
    '''
    if len(affected_boundaries) == 0:
        z_scores = [0 for x in np.arange(0, len(boundaries), 1)]
        affected_boundaries = boundaries.copy()
    bounds = [0]
    for node in boundaries:
        bounds.append(node)
    bounds.append(len(sequence))
    raw_scores = get_region_scores(sequence, boundaries)
    for boundary in affected_boundaries:
        ## The two modules separated by a boundary are combined into one 'region' ##
        n = bounds.index(boundary)
        region = [sequence[int(bounds[n - 1]):int(bounds[n + 1])], np.min(
            [bounds[n] - bounds[n - 1], bounds[n + 1] - bounds[n]]), raw_scores[n - 1]]
        z_set = list()
        count = 0
        while count < 500:
            scramble = ''.join(random.sample(region[0], len(region[0])))
            scores = list()
            for x in np.arange(region[1], len(region[0]) - (region[1] - 1), 1):
                scores.append(calculate_chi_score(
                    scramble[0:int(x)], scramble[int(x):]))
            z_set.append(np.max(scores))
            count = count + 1
        if np.std(z_set) != 0:
            z_scores[n - 1] = round((region[2] -
                                    np.mean(z_set)) / np.std(z_set), 3)
        else:
            z_scores[n - 1] = 0
    return z_scores


def trim_boundaries(sequence, initial_solution, initial_zscores, boundary_groups, cutoff=1.96):
    ''' 
    Trims boundaries in solution by iteratively removing those with low z-scores and reoptimizing those that remain
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    initial_solution: list of optimized boundary placements before trimming begins
    initial_zscores: list of initial z-scores for initial boundaries
    boundary_groups: list of groups from which each bounadry's position can be optimized
    cutoff: this function finishes when all remaining boundaries have z-scores above this value; a z-score of 1.96 corresponds to a confidence level of 95%
    Outputs list of solutions (optimized boundary positions after each iteration) and list of corresponding z-scores
    '''
    trimmed_solutions = [initial_solution.copy()]
    trimmed_zscores = [initial_zscores.copy()]
    min_zscore = np.min(initial_zscores)
    considered_groups = boundary_groups.copy()
    while min_zscore < cutoff:
        n = trimmed_zscores[-1].index(min_zscore)
        new_groups = list()
        new_zscores = list()
        for x in np.arange(0, len(considered_groups), 1):
            if trimmed_zscores[-1][x] != np.min(trimmed_zscores[-1]):
                new_groups.append(considered_groups[x].copy())
                new_zscores.append(trimmed_zscores[-1][x])
        if len(new_groups) > 0:
            considered_groups = new_groups.copy()
            new_solution = optimize_boundaries(sequence, considered_groups)
            new_positions = list()
            for x in np.arange(0, len(new_solution) + 1, 1):
                if x < n:
                    new_positions.append(new_solution[x])
                elif x == n:
                    new_positions.append(0)
                elif x > n:
                    new_positions.append(new_solution[x - 1])
            affected_boundaries = list()
            for x in np.arange(0, len(new_positions), 1):
                if x == 0:
                    if new_positions[x] != trimmed_solutions[-1][x] or new_positions[x + 1] != trimmed_solutions[-1][x + 1]:
                        if new_positions[x] != 0:
                            affected_boundaries.append(new_positions[x])
                elif x == len(new_positions) - 1:
                    if new_positions[x] != trimmed_solutions[-1][x] or new_positions[x - 1] != trimmed_solutions[-1][x - 1]:
                        if new_positions[x] != 0:
                            affected_boundaries.append(new_positions[x])
                else:
                    if new_positions[x] != trimmed_solutions[-1][x] or new_positions[x + 1] != trimmed_solutions[-1][x + 1] or new_positions[x - 1] != trimmed_solutions[-1][x - 1]:
                        if new_positions[x] != 0:
                            affected_boundaries.append(new_positions[x])
            trimmed_solutions.append(new_solution.copy())
            new_zscores = get_zscores(
                sequence, trimmed_solutions[-1], affected_boundaries, new_zscores)
            trimmed_zscores.append(new_zscores.copy())
            min_zscore = np.min(new_zscores)
        else:
            min_zscore = 3.0
    return trimmed_solutions, trimmed_zscores


def analyze_sequence(sequence, window_sizes=[6, 8, 10, 12, 14, 16, 18, 20, 22], groups='twenty'):
    ''' 
    Performs the complete Chi-Score Analysis on input sequence from pairwise matrix generation to z-score validation
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    window_sizes: list of integers corresponding to which subsequence lengths to predict boundary positions with; even integers recommended
    groups: amino acid grouping scheme to use, see 'residue groups' for list of grouping schemes
    Outputs list of solutions, each containing a list of boundary positions and a list of corresponding z-scores
    '''
    window_groups = list()
    heatmap_scores = list()
    print(
        f'Now calculating boundaries for window sizes {window_sizes[0]} through {window_sizes[-1]}')
    for window_size in window_sizes:
        ## For each window size, pairwise matrices, insulation scores, and initial boundary placements are determined ##
        raw_scores = get_heatmap_scores(sequence, window_size, groups)
        corr_scores = get_corr_scores(raw_scores)
        insulation_scores = get_insulation_scores(corr_scores, (2*window_size)-1)
        boundaries = subsequence_to_boundary(get_minima(insulation_scores), window_size)
        heatmap_scores.append(corr_scores)
        window_groups.append(boundaries)
    print('Now clustering boundaries and determining initial solution.')
    boundary_groups = cluster_boundaries(window_groups)
    bgs = boundary_groups.copy()
    initial_solution = optimize_boundaries(sequence, boundary_groups)
    module_lengths = list()
    module_lengths.append(initial_solution[0])
    for x in np.arange(1, len(initial_solution), 1):
        module_lengths.append(initial_solution[x] - initial_solution[x - 1])
    if np.min(module_lengths) < 6:
        print(
            'Short modules found in initial solution. Now merging relevant boundary groups.')
        initial_solution, boundary_groups = eliminate_short_modules(
            sequence, initial_solution, boundary_groups)
    print('Now calculating z-scores and trimming low-confidence boundaries.')
    initial_zscores = get_zscores(sequence, initial_solution)
    trimmed_solutions, trimmed_zscores = trim_boundaries(
        sequence, initial_solution, initial_zscores, boundary_groups)
    solutions = [[initial_solution, initial_zscores]]
    for x in np.arange(0, len(trimmed_solutions), 1):
        solutions.append([trimmed_solutions[x], trimmed_zscores[x]])
    return solutions


def try_analysis(sequence, groups='twenty'):
    ''' 
    Function that tries the full Chi-Score Analysis on input sequence and assumes no modules were found in sequence fails.
    Useful for analyzing large sets of sequences so that sequences that fail to be analyzed (for whatever reason) don't hault the process.
    '''
    try:
        results = analyze_sequence(sequence, groups=groups)
    except:
        results = 'No Modules Found in Sequence'
    finally:
        return results

def plot_solution(sequence, corr_scores, solution, window = 12, name = 'input sequence'):
    '''
    Visualizes results of analysis for input sequence; includes correlated matrix, bounadry positions, and local complexity plots
    sequence: string of capitalized amino acid characters (ex: 'MSTAVG...')
    corr_scores: matrix of pairwise subsequence correlation coefficients to be rendered
    solution: bounadary positions to be plotted over matrix
    window: window size used to generate corr_scores, required to plot solutions correctly over matrix
    name: name of the sequence being visualized; input as a string, this will be included in the title of the output figure
    Outputs Pyplot matrix of pairwise correlation coefficients with the identified boundaries plotted as vertical lines; above the matrix is a plot of local sequence complexity
    '''
    local_complexities = get_complexity(sequence, window)
    fig, ([ax0, ax2], [ax1, ax3]) = plt.subplots(nrows = 2, ncols = 2, sharex = True, gridspec_kw={'height_ratios': [0.75, 7], 'width_ratios': [7, 1]}, figsize = [8, 7.75])
    ax0.set_title('Complexity and Correlated Heatmap for {}'.format(name))
    ax0.axhline(np.mean(local_complexities), color = 'black', alpha = 0.5)
    ax0.set(ylim=(-0.1, 3.1))
    ax0.axis('off')
    hm = ax1.imshow(corr_scores, cmap = 'Blues', norm=plt.Normalize(-1, 1))
    slen = len(sequence)
    nwin = slen - (window - 1)
    labels = list()
    for x in np.arange(((window-1)/2), slen-((window-1)/2), step = int(len(sequence)/5)):
        labels.append(int(x))
    ax1.set_xticks(np.arange(0, nwin, step = int(len(sequence)/5)))
    ax1.set_xticklabels(labels)
    ax1.set_yticks(np.arange(0, nwin, step = int(len(sequence)/5)))
    ax1.set_yticklabels(labels)
    for x in np.arange(0, len(solution), 1):
        node_spot = solution[x]-((window-1)/2)
        ax1.axvline(node_spot, color = 'r', alpha = 1)
    ax1.set_xlabel('Residue')
    ax1.set_ylabel('Residue')
    x = np.linspace(0, len(sequence)-(window-1), len(sequence)-(window-1))
    y = np.asarray(local_complexities)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(-0.1, 3.1)
    lc = LineCollection(segments, cmap='inferno', norm=norm)
    lc.set_array(y)
    lc.set_linewidth(1)
    line = ax0.add_collection(lc)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax0)
    cax2 = divider2.append_axes("right", size="5%", pad = 0.1)
    cbar0 = plt.colorbar(hm, cax=cax)
    cbar1 = plt.colorbar(line, cax=cax2, ticks=[round(y.min(), 2), round(y.max(), 2)])
    cbar1.ax.set_ylabel('SHANNON\nENTROPY', rotation = 0, labelpad=5, fontsize='x-small', y=0.7, ha='left')
    cbar0.ax.set_ylabel('PEARSON\nCORRELATION\nCOEFFICIENT', rotation=0, labelpad=-5, fontsize='x-small', y=0.55, ha='left')
    ax2.axis('off')
    ax3.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
