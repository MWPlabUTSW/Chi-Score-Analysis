## Benchmark Job for BioHPC Test ##

import random
import timeit
import numpy as np
import pandas as pd


def count_residues_in_sequence(sequence, residue):
    residue_count = 0
    residue_count += sequence.count(residue)
    return residue_count


def get_observed_values(sequence_1, sequence_2, groups='twenty'):
    obs = np.zeros((2, len(residue_groups[groups])))
    fracsall1 = list()
    for group in residue_groups[groups]:
        count = 0
        for residue in group:
            count += count_residues_in_sequence(sequence_1, residue)
        fraction = count / len(sequence_1)
        fracsall1.append(fraction * 1000)
    obs[0] = fracsall1
    fracsall2 = list()
    for group in residue_groups[groups]:
        count = 0
        for residue in group:
            count += count_residues_in_sequence(sequence_2, residue)
        fraction = count / len(sequence_2)
        fracsall2.append(fraction * 1000)
    obs[1] = fracsall2
    return obs


def get_expected_values(observed_values):
    exp = np.zeros((2, len(observed_values[0])))
    for seqs in [0, 1]:
        for group in range(0, len(observed_values[0])):
            exp[seqs, group] = (observed_values[0, group] +
                                observed_values[1, group]) * 0.5
    return exp


def get_residue_conts(observed_values, expected_values):
    conts = np.zeros((2, len(observed_values[0])))
    for seqs in [0, 1]:
        for group in range(0, len(observed_values[0])):
            if expected_values[seqs, group] == 0:
                conts[seqs, group] = 0
            else:
                conts[seqs, group] = (
                    (observed_values[seqs, group] - expected_values[seqs, group])**2) / expected_values[seqs, group]
    conts_t = conts.transpose()
    res_conts = np.zeros((1, len(observed_values[0])))
    for group in range(0, len(observed_values[0])):
        res_conts[0, group] = sum(conts_t[group, :])
    return res_conts


def get_chi_score(residue_conts):
    return residue_conts.sum() / 2000


def calculate_chi_score(sequence_1, sequence_2, groups='twenty'):
    obs = get_observed_values(sequence_1, sequence_2, groups)
    exp = get_expected_values(obs)
    res_conts = get_residue_conts(obs, exp)
    return get_chi_score(res_conts)


def get_heatmap_scores(sequence, window, groups='twenty'):
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
    return(scores)


def get_corr_scores(scores):
    df = pd.DataFrame(scores)
    dfcorr = df.corr()
    corr_scores = dfcorr.to_numpy()
    return corr_scores

# New Instulation Score Definition


def ins_to_residues(ins_numbers, chi_window):
    residues = list()
    for ins in ins_numbers:
        residues.append(ins + ((chi_window - 1) / 2))
    return residues


def get_region_scores(sequence, optimized_nodes):
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
    bounds = [0]
    for bound in boundaries:
        bounds.append(bound)
    bounds.append(len(sequence))
    modules = list()
    for x in np.arange(1, len(bounds), 1):
        modules.append(sequence[int(bounds[x - 1]):int(bounds[x])])
    return modules


def optimize_boundaries(sequence, groups):
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


def get_insulation_scores(corr_scores, ins_window=10):
    window_range = np.arange(ins_window, len(corr_scores) - (ins_window), 1)
    insulation_scores = list()
    for x in window_range:
        windowa = list()
        windowb = list()
        if x == window_range[0]:
            for y in np.arange(1, ins_window + 1, 1):
                windowa.append(corr_scores[x - y, x + y])
            insulation_scores.append(np.mean(windowa))
        else:
            for y in np.arange(0, ins_window, 1):
                windowa.append(corr_scores[x - y - 1, x + y])
            insulation_scores.append(np.mean(windowa))
            for y in np.arange(1, ins_window + 1, 1):
                windowb.append(corr_scores[x - y, x + y])
            insulation_scores.append(np.mean(windowb))
    smoothed_scores = list()
    for x in np.arange(0, len(insulation_scores), 1):
        if x == 0:
            smoothed_scores.append(np.mean(insulation_scores[0:2]))
        elif x == len(insulation_scores) - 1:
            smoothed_scores.append(np.mean(insulation_scores[-2:]))
        else:
            smoothed_scores.append(np.mean(insulation_scores[x - 1:x + 2]))
    return smoothed_scores


def get_minima(insulation_scores, corr_scores, window_size):
    xs = np.arange(10, len(corr_scores) - (10 + 0.5), 0.5)
    minima = list()
    min_scores = list()
    for x in np.arange(1, len(insulation_scores) - 1, 1):
        if insulation_scores[x] < insulation_scores[x - 1] and insulation_scores[x] < insulation_scores[x + 1] and insulation_scores[x] < np.mean(insulation_scores):
            minima.append(xs[x])
            min_scores.append(insulation_scores[x])
    adjusted_minima = ins_to_residues(minima, window_size)
    min_array = np.zeros((len(minima), 2))
    for x in np.arange(0, len(minima), 1):
        min_array[x, 0], min_array[x, 1] = adjusted_minima[x], min_scores[x]
    min_array = min_array[min_array[:, 1].argsort()]
    solution = list()
    blacklist = list()
    for row in min_array:
        if row[0] not in blacklist:
            solution.append(row[0])
            black_range = np.arange(row[0] - 9.5, row[0] + (10), 0.5)
            for residue in black_range:
                blacklist.append(residue)
    solution.sort()
    return solution


def cluster_boundaries(window_groups):
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
                if np.min(diffs) >= 5:
                    boundary_groups.append([node])
                else:
                    node_index = np.column_stack(
                        np.where(diffs == np.min(diffs)))
                    boundary_groups[node_index[0][0]].append(node)
    sorted_groups = list()
    for group in boundary_groups:
        new_group = list()
        for minimum in group:
            nodes = list()
            if int(minimum) == minimum:
                nodes.append(minimum)
            else:
                nodes.append(minimum - 0.5)
                nodes.append(minimum + 0.5)
            for node in nodes:
                if node not in new_group:
                    new_group.append(node)
        new_group.sort()
        sorted_groups.append(new_group)
    sorted_groups.sort()
    return sorted_groups


def eliminate_short_modules(sequence, initial_solution, boundary_groups, cutoff=6):
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
    if len(affected_boundaries) == 0:
        z_scores = [0 for x in np.arange(0, len(boundaries), 1)]
        affected_boundaries = boundaries.copy()
    bounds = [0]
    for node in boundaries:
        bounds.append(node)
    bounds.append(len(sequence))
    raw_scores = get_region_scores(sequence, boundaries)
    for boundary in affected_boundaries:
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
        z_scores[n - 1] = round((region[2] -
                                 np.mean(z_set)) / np.std(z_set), 3)
    return z_scores


def trim_boundaries(sequence, initial_solution, initial_zscores, boundary_groups, cutoff=2.58):
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


def analyze_sequence(sequence, window_sizes=[5, 7, 9, 11, 13, 15, 17, 19, 21], groups='twenty'):
    window_groups = list()
    heatmap_scores = list()
    print(
        f'Now calculating boundaries for window sizes {window_sizes[0]} through {window_sizes[-1]}')
    for window_size in window_sizes:
        raw_scores = get_heatmap_scores(sequence, window_size, groups)
        corr_scores = get_corr_scores(raw_scores)
        insulation_scores = get_insulation_scores(corr_scores, 10)
        boundaries = get_minima(insulation_scores, corr_scores, window_size)
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
    return solutions, heatmap_scores


random.seed(0)

props = [0.10769882239070817,
         0.0564607194708824,
         0.04282948862719794,
         0.04242619777383449,
         0.013873205355702533,
         0.05210517825455719,
         0.0503508630424262,
         0.06702693982900468,
         0.021475237941603484,
         0.039199870946926924,
         0.07900467817389902,
         0.05859816099370866,
         0.03831263106952734,
         0.02893611872882723,
         0.06573640909824165,
         0.08991369575738022,
         0.0667849653169866,
         0.007904500725923537,
         0.015183900629133731,
         0.05617841587352799]
residues = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

lens = [50, 100, 200, 500, 1000]

runtimes = list()
results = list()

for x in lens:
    seqs = [''.join(random.choices(residues, props, k=x)) for y in range(5)]
    for seq in seqs:
        start_time = timeit.default_timer()
        sols, scores = analyze_sequence(seq)
        runtime = timeit.default_timer() - start_time
        results.append([sols, scores])
        runtimes.append(runtime)

print(runtimes)
