
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
    ## Counts number of 'residue' in 'sequence' ##
    residue_count = 0
    residue_count += sequence.count(residue)
    return residue_count


def get_observed_values(sequence_1, sequence_2, groups='twenty'):
    ## Computes observed values for chi-score calculation for amino acid groups in compared sequences ##
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
    ## Computes expected values from observed values ##
    exp = np.zeros((2, len(observed_values[0])))
    for seqs in [0, 1]:
        for group in range(0, len(observed_values[0])):
            exp[seqs, group] = (observed_values[0, group] +
                                observed_values[1, group]) * ratio
    return exp


def get_chi_score(observed_values, expected_values, res_conts=False):
    ## Computes chi-score from observed and expected values ##
    chi_scores = np.zeros((2, len(observed_values[0])))
    for seqs in [0, 1]:
        for group in range(len(observed_values[0])):
            if expected_values[seqs, group] != 0:
                chi_scores[seqs, group] = (
                    (observed_values[seqs, group] - expected_values[seqs, group])**2) / expected_values[seqs, group]
            else:
                chi_scores[seqs, group] = 0
    ## If res_conts is True, returns list of the contributions of each residue group ##
    if res_conts is True:
        return [np.sum(chi_scores[:, group]) / np.sum(observed_values) for group in range(len(observed_values[0]))]
    else:
        return np.sum(chi_scores) / np.sum(observed_values)


def calculate_chi_score(sequence_1, sequence_2, groups='twenty'):
    ## Combines above functions into one that computes chi-score between two sequences ##
    obs = get_observed_values(sequence_1, sequence_2, groups)
    exp = get_expected_values(obs)
    return get_chi_score(obs, exp)


def get_heatmap_scores(sequence, window, groups='twenty'):
    slen = len(sequence)
    nwin = slen - (window - 1)
    seqs = list()
    ## Define all subsequences in sequence ##
    for windows in range(0, nwin):
        subseq = sequence[windows:windows + window]
        seqs.append(subseq)
    ## Compute pairwise chi-scores between subsequences ##
    scores = np.zeros((nwin, nwin))
    for x in range(0, nwin):
        for y in range(0, nwin):
            scores[x, y] = calculate_chi_score(seqs[x], seqs[y], groups)
    ## Returns matrix of pairwise scores ##
    return scores


def get_corr_scores(scores):
    ## Performs Pearson Correlation on pairwise chi-score matrix ##
    df = pd.DataFrame(scores)
    dfcorr = df.corr()
    corr_scores = dfcorr.to_numpy()
    ## Returns matrix of correlation coefficients ##
    return corr_scores


def get_insulation_scores(corr_matrix, s):
    ## Computes insulation scores from matrix of correlation coefficients in square window of size 's' ##
    n_scores = len(corr_matrix)
    ## Define square windows from which scores are calculated ##
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
    ## Returns list of scores, one for each subsequence in the matrix ##
    return [np.mean(window) for window in windows]


def get_minima(insulation_scores):
    ## Identifies local minima (potential boundaries) from insulation scores ##
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
    ## Excessively close minima rejected by excluding those within 10 residues of lowest scoring minimum ##
    solution = list()
    blacklist = list()
    for row in min_array:
        if row[0] not in blacklist:
            solution.append(row[0])
            black_range = np.arange(row[0] - 9, row[0] + 10, 1)
            for residue in black_range:
                blacklist.append(residue)
    solution.sort()
    ## Returns list of positions (x-values) corresponding to minima ##
    return solution


def subsequence_to_boundary(subsequence_indices, subsequence_length):
    ## Converts subsequence indices to residue indices (where boundary actually occurs) ##
    boundary_indices = list()
    for index in subsequence_indices:
        boundary_indices.append(index + (subsequence_length / 2))
    return boundary_indices


def get_region_scores(sequence, optimized_nodes):
    ## Computes chi-scores between neighboring regions (separated by a node) ##
    boundaries = [0]
    for node in optimized_nodes:
        if int(node) not in boundaries:
            boundaries.append(int(node))
    boundaries.append(len(sequence))
    if len(boundaries) == 2:
        return 0
    else:
        ## Defines modules from boundary positions ##
        nregions = len(boundaries) - 1
        regions = list()
        for region in range(0, nregions):
            regions.append(sequence[boundaries[region]:boundaries[region + 1]])
        scores = list()
        for x in range(0, nregions):
            ## Computes chi-score between the two modules separated by each boundary ##
            for y in range(0, nregions):
                if y == x - 1:
                    if len(regions[x]) < 1 or len(regions[y]) < 1:
                        scores.append(0)
                    else:
                        scores.append(calculate_chi_score(
                            regions[x], regions[y]))
        ## Returns list of chi-scores corresponding to boundaries in 'optimized_nodes' ##
        return scores


def get_modules(sequence, boundaries):
    ## Returns amino acid sequences of modules in 'sequence' as defined by 'boundaries' ##
    bounds = [0]
    for bound in boundaries:
        bounds.append(bound)
    bounds.append(len(sequence))
    modules = list()
    for x in np.arange(1, len(bounds), 1):
        modules.append(sequence[int(bounds[x - 1]):int(bounds[x])])
    return modules


def optimize_boundaries(sequence, groups):
    ## Determines the optimal placement for each boundary in 'groups' for 'sequence' ##
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
    ## From left to right, three consecutve boundary positions are varied at a time, with the central position optimized in each step ##
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
    ## Returns single list of optimized boundary placements ##
    return optimized_sols


def get_complexity(sequence, window):
    ## Computes local sequence complexities as Shannon Entropy in 'sequence' with specified 'window' size ##
    nwin = len(sequence) - (window - 1)
    ## Define all subsequences of length 'window' ##
    subseqs = list()
    for windows in np.arange(0, nwin, 1):
        subseq = sequence[windows:windows + window]
        subseqs.append(subseq)
    complexities = list()
    for seq in subseqs:
        pk = list()
        ## Determine probability distribution of each residue in subseqeunce ##
        for res in ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']:
            pk.append(count_residues_in_sequence(seq, res) / len(seq))
        ## Compute Shannon Entropy from probability distribution ##
        complexities.append(entropy(pk))
    ## Returns list of complexity scores ##
    return complexities


def cluster_boundaries(window_groups):
    ## Changes grouping of 'window_groups' from which window size was used to residue proximity ##
    ## Input list is grouped by which window size was used to determine placements ##
    boundaries0 = window_groups[0].copy()
    ## Initial boundary groups defined by first group in 'window_groups' ##
    boundary_groups = list()
    for boundary in boundaries0:
        boundary_groups.append([boundary])
    ## In the other 'window_groups', each residue is clustered with its nearest boundary group that is already defined ##
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
    ## Each new group is sorted from low to high boundary positions ##
    sorted_groups = list()
    for group in boundary_groups:
        new_group = list()
        for node in group:
            new_group.append(node)
        new_group.sort()
        sorted_groups.append(new_group)
    sorted_groups.sort()
    ## Returns list of boundary groups, with each containing all potential placements for that boundary ##
    return sorted_groups


def eliminate_short_modules(sequence, initial_solution, boundary_groups, cutoff=6):
    ## Iteratively removes boundaries that result in modules shorter than the specified 'cutoff' ##
    module_lengths = list()
    module_lengths.append(initial_solution[0])
    for x in np.arange(1, len(initial_solution), 1):
        module_lengths.append(initial_solution[x] - initial_solution[x - 1])
    shortest_mod = np.min(module_lengths)
    considered_groups = list()
    for group in boundary_groups:
        considered_groups.append(group.copy())
    temporary_solution = initial_solution.copy()
    ## If solution has module shorter than the 'cutoff', the two boundary groups are merged into one boundary, whose postiion is then reoptimized ##
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
    ## Returns updated solution (optimized) and remaining boundary groups (unoptimized) ##
    return temporary_solution.copy(), considered_groups.copy()


def get_zscores(sequence, boundaries, affected_boundaries=[], z_scores=[]):
    ## Computes the z-scores for each boundary in 'boundaries' in 'sequence' ##
    ## 'affected_boundaries' denotes which z-scores may now be different and need computed ##
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
            ## Each region is scrambled 500 times, and the boundary with the highest chi-score is identified in each ##
            scramble = ''.join(random.sample(region[0], len(region[0])))
            scores = list()
            for x in np.arange(region[1], len(region[0]) - (region[1] - 1), 1):
                scores.append(calculate_chi_score(
                    scramble[0:int(x)], scramble[int(x):]))
            z_set.append(np.max(scores))
            count = count + 1
        ## Mean and standard deviation of random scores used to calculate boundary z-scores ##
        if np.std(z_set) != 0:
            z_scores[n - 1] = round((region[2] -
                                    np.mean(z_set)) / np.std(z_set), 3)
        else:
            z_scores[n - 1] = 0
    ## Returns list of z-scores corresponding to each module boundary ##
    return z_scores


def trim_boundaries(sequence, initial_solution, initial_zscores, boundary_groups, cutoff=1.96):
    ## Iteratively removes boundary groups, reoptimizes those that remain, and comptues their z-scores until all remaining boundaries have z-scores abot the 'cutoff' ##
    trimmed_solutions = [initial_solution.copy()]
    trimmed_zscores = [initial_zscores.copy()]
    min_zscore = np.min(initial_zscores)
    considered_groups = boundary_groups.copy()
    ## If lowest z-score is lower than cutoff, that boundary group is removed and the optimization/scoring step is repeated ##
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
            ## Optimized positions and their z-scores are saved after each iteration to be recalled as desired ##
            trimmed_solutions.append(new_solution.copy())
            new_zscores = get_zscores(
                sequence, trimmed_solutions[-1], affected_boundaries, new_zscores)
            trimmed_zscores.append(new_zscores.copy())
            min_zscore = np.min(new_zscores)
        else:
            min_zscore = 3.0
    ## Process repeats until all z-scores are above 'cutoff' or no boundary groups remain ##
    ## Outputs list of solutions and list of corresponding z-scores for each iteration ##
    return trimmed_solutions, trimmed_zscores


def analyze_sequence(sequence, window_sizes=[6, 8, 10, 12, 14, 16, 18, 20, 22], groups='twenty'):
    ## Combines analysis functions into one that analyzes a 'sequence' with specified 'window_sizes' and runs whole analysis ##
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
    ## Boundary groups are clustered by residue proximity and initially optimized ##
    print('Now clustering boundaries and determining initial solution.')
    boundary_groups = cluster_boundaries(window_groups)
    bgs = boundary_groups.copy()
    initial_solution = optimize_boundaries(sequence, boundary_groups)
    ## Short modules are eliminated from initial solution ##
    module_lengths = list()
    module_lengths.append(initial_solution[0])
    for x in np.arange(1, len(initial_solution), 1):
        module_lengths.append(initial_solution[x] - initial_solution[x - 1])
    if np.min(module_lengths) < 6:
        print(
            'Short modules found in initial solution. Now merging relevant boundary groups.')
        initial_solution, boundary_groups = eliminate_short_modules(
            sequence, initial_solution, boundary_groups)
    ## Z-Scores are calculated and iterative trimming process is performed ##
    print('Now calculating z-scores and trimming low-confidence boundaries.')
    initial_zscores = get_zscores(sequence, initial_solution)
    trimmed_solutions, trimmed_zscores = trim_boundaries(
        sequence, initial_solution, initial_zscores, boundary_groups)
    solutions = [[initial_solution, initial_zscores]]
    for x in np.arange(0, len(trimmed_solutions), 1):
        solutions.append([trimmed_solutions[x], trimmed_zscores[x]])
    ## Outputs list of solutions, each containing boundary placements and corresponding z-scores ##
    return solutions


def try_analysis(sequence, groups=residue_groups['twenty']):
    ## Function that attempts analysis on sequence and moves on if failed ##
    ## Useful for running long lists of sequences where some might fail for various reasons ##
    try:
        results = analyze_sequence(sequence, groups=groups)
    except:
        results = 'No Modules Found in Sequence'
    finally:
        return results

def plot_solution(sequence, corr_scores, solution, window = 11, name = 'input sequence', outfile=False):
    ## Plots 'corr_scores' matrix ad local complexity for 'sequence' and maps 'solution' as vertical lines ##
    ## If specified, saves figure to outfile ##
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
    # Add in color bars for both heatmap and complexity plot #
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    divider2 = make_axes_locatable(ax0)
    cax2 = divider2.append_axes("right", size="5%", pad = 0.1)
    cbar0 = plt.colorbar(hm, cax=cax)
    cbar1 = plt.colorbar(line, cax=cax2, ticks=[round(y.min(), 2), round(y.max(), 2)])
    cbar1.ax.set_ylabel('SHANNON\nENTROPY', rotation = 0, labelpad=5, fontsize='x-small', y=0.7, ha='left')
    cbar0.ax.set_ylabel('PEARSON\nCORRELATION\nCOEFFICIENT', rotation=0, labelpad=-5, fontsize='x-small', y=0.55, ha='left')
    # Hide extra subplots and remove empty space #
    ax2.axis('off')
    ax3.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
