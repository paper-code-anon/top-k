import pandas as pd
import numpy as np
import scipy.optimize


# Declare private function for convenience use by top_k_algorithm_1
def __get_cutoff_at_perp(df, v_perp_id):
    return [k for k, v in enumerate(df.index.values) if v == v_perp_id][0]


# get_func_0:4 are used for creating the equation to be optimized for the epsilon breakdown
def get_func_0(k, eps_g):
    def func_0(x):
        return x[0] - eps_g / (2 * k)

    return func_0


def get_func_1(k, eps_g, little_delta_1):
    def func_1(x):
        return k * x[0] * (np.exp(x[0]) - 1) / (np.exp(x[0]) + 1) + \
               x[0] * np.sqrt(2 * k * np.log(1 / little_delta_1)) - \
               eps_g / 2

    return func_1


def get_func_2(k, eps_g, little_delta_1):
    def func_2(x):
        return (k / 2) * (x[0] ** 2) + \
               x[0] * np.sqrt(0.5 * k * np.log(1 / little_delta_1)) - \
               eps_g / 2

    return func_2


def get_func_3(k, eps_g, little_delta):
    def func_3(x):
        return k * x[0] * (np.exp(x[0]) - 1) + \
               x[0] * np.sqrt(2 * k * np.log(1 / little_delta)) - \
               eps_g / 2

    return func_3


def get_eps_1_2_algorithm_1(epsilon_g, little_delta_1, little_delta_2, k):
    initial_guess = 1  # Solver requires an input for starting value

    # Calculate potential values for epsilon_1
    epsilon_1_0 = scipy.optimize.broyden1(get_func_0(k, epsilon_g), [initial_guess], f_tol=1e-14)[0]
    epsilon_1_1 = scipy.optimize.broyden1(get_func_1(k, epsilon_g, little_delta_1), [initial_guess], f_tol=1e-14)[0]
    epsilon_1_2 = scipy.optimize.broyden1(get_func_2(k, epsilon_g, little_delta_1), [initial_guess], f_tol=1e-14)[0]
    epsilon_1 = max(epsilon_1_0, epsilon_1_1, epsilon_1_2)

    # Calculate potential values for epsilon_2
    epsilons_2_0 = scipy.optimize.broyden1(get_func_0(k, epsilon_g), [initial_guess], f_tol=1e-14)[0]
    epsilons_2_1 = scipy.optimize.broyden1(get_func_3(k, epsilon_g, little_delta_2), [initial_guess], f_tol=1e-14)[0]
    epsilon_2 = max(epsilons_2_0, epsilons_2_1)

    return epsilon_1, epsilon_2


def get_eps_1_2_algorithm_2(epsilon_g, little_delta_1, little_delta_2, k):
    initial_guess = 1  # Solver requires an input for starting value

    # Calculate potential values for epsilon_1
    epsilon_1_0 = scipy.optimize.broyden1(get_func_0(k, epsilon_g), [initial_guess], f_tol=1e-14)[0][0]
    epsilon_1_1 = scipy.optimize.broyden1(get_func_3(k, epsilon_g, little_delta_1), [initial_guess], f_tol=1e-14)[0]
    epsilon_1 = max(epsilon_1_0, epsilon_1_1)

    # Calculate potential values for epsilon_2
    epsilons_2_0 = scipy.optimize.broyden1(get_func_0(k, epsilon_g), [initial_guess], f_tol=1e-14)[0]
    epsilons_2_1 = scipy.optimize.broyden1(get_func_3(k, epsilon_g, little_delta_2), [initial_guess], f_tol=1e-14)[0]
    epsilon_2 = max(epsilons_2_0, epsilons_2_1)

    return epsilon_1, epsilon_2


def top_k_algorithm_1(df_histo, epsilon_g, little_delta, little_delta_p, big_delta, k, k_bar):
    # 0) Calculate the optimal breakdown of epsilon between algorithm steps using equation TODO: Ref
    epsilon_1, epsilon_2 = get_eps_1_2_algorithm_1(epsilon_g, little_delta, little_delta, k)

    # 1) Calculate h_perp
    h_perp = df_histo.iloc[k_bar] + 1 + np.log(min(big_delta, k_bar) / little_delta_p) / epsilon_1

    # 2) Calculate v_perp
    v_perp = h_perp + np.random.gumbel(0, 1 / epsilon_1, None)

    # 3) Cut the histogram at length k_bar
    df_histo_cut = df_histo.iloc[0: k_bar]

    # 4) Apply gumbel noise to each element of the histogram
    df_histo_pert = df_histo_cut + np.random.gumbel(0, 1 / epsilon_1, df_histo_cut.size)

    # 5) Append v_perp to the perturbed histogram
    v_perp_index_name = 'v_perp'
    df_histo_pert_unsorted = df_histo_pert.append(pd.Series({v_perp_index_name: v_perp}))

    # 6) Sort perturbed histogram
    df_histo_pert_sorted = df_histo_pert_unsorted.sort_values(ascending=False)

    # 7) Find the index j, for the sorted list up until v_perp
    j = __get_cutoff_at_perp(df_histo_pert_sorted, v_perp_index_name)

    # 7.5) Apply round of laplace noise to the original frequencies, keys selected from df_histo_pert_sorted
    df_pert_selected_original_frequencies = df_histo_cut.loc[df_histo_pert_sorted.index.drop('v_perp')]
    # Apply noise to the original frequencies
    df_pert_selected_original_frequencies_pert = df_pert_selected_original_frequencies + \
        np.random.laplace(0, 1 / epsilon_2, df_pert_selected_original_frequencies.size)

    if j < k:
        # Include v_perp as the final element
        return df_pert_selected_original_frequencies_pert.iloc[0: j].append(pd.Series({v_perp_index_name: v_perp})), v_perp, epsilon_1, epsilon_2
    else:
        return df_pert_selected_original_frequencies_pert.iloc[0: k], v_perp, epsilon_1, epsilon_2


def top_k_algorithm_2(df_histo, epsilon_g, little_delta, k):
    # 0) Calculate the optimal breakdown of epsilon between algorithm steps using equation TODO: Ref
    epsilon_1, epsilon_2 = get_eps_1_2_algorithm_1(epsilon_g, little_delta, little_delta, k)

    # 1) Apply laplace noise to each element of the histogram
    df_histo_pert = df_histo + np.random.laplace(0, 1 / epsilon_1, df_histo.size)

    # 2) Select top k of perturbed histogram
    pert_top_k = df_histo_pert.sort_values(ascending=False).iloc[0: k]

    # 3) Apply round of laplace noise to the original frequencies, selected from the top k of perturbed frequencies
    pert_top_k_original_frequencies = df_histo.loc[pert_top_k.index]
    # Apply noise to the original frequencies
    original_pert_top_k = pert_top_k_original_frequencies + np.random.laplace(0, 1 / epsilon_2,
                                                                              pert_top_k_original_frequencies.size)

    # 4) Floor any values at 0
    original_pert_top_k_floored = original_pert_top_k.map(lambda x: 0 if x < 0 else x)

    return original_pert_top_k_floored.sort_values(ascending=False), epsilon_1, epsilon_2


def get_next_value_below_cutoff(df_histo, k, percentile_cutoff):
    kth_val = df_histo.iloc[k - 1]
    frequency_cutoff = kth_val * percentile_cutoff / 100
    reverse_index = np.searchsorted(np.flip(df_histo.values), frequency_cutoff, side='right')
    if reverse_index == 0:
        raise ValueError('k_bar is out of range, please use a larger percentile_cutoff')
    index = len(df_histo) - reverse_index
    return index


def calculate_metrics(original_data,
                      perturbed_top_k,
                      algorithm_type,
                      epsilon_g,
                      epsilon_1,
                      epsilon_2,
                      little_delta_prime,
                      beta,
                      k,
                      k_bar=None,
                      d=None,
                      percentile_cutoff=None):
    if algorithm_type == 1:
        y = (1 / epsilon_1) * np.log(k * k_bar / beta)
        eta = (1 / epsilon_2) * np.log(k / beta)
        previous_epsilon = 1 / ((np.sqrt(0.5 * k * np.log(1 / little_delta_prime))) / (epsilon_g / 2))
    elif algorithm_type == 2:
        y = (2 / epsilon_1) * np.log(d / beta)
        eta = (1 / epsilon_2) * np.log(k / beta)
        previous_epsilon = 1 / ((2 / epsilon_g) * np.sqrt(8 * k * np.log(1 / little_delta_prime)))
    else:
        ValueError('Invalid algorithm type')

    # Set the indexes for ease of calculation
    original_data_df = pd.DataFrame(original_data)
    perturbed_top_k_df = pd.DataFrame(perturbed_top_k)
    original_data_df.columns = ['Frequency']
    perturbed_top_k_df.columns = ['Frequency']
    original_vs_perturbed_df = perturbed_top_k_df.join(original_data_df, how='outer', rsuffix='_original')

    # Calculate f_k
    f_k = np.flip(np.sort(original_vs_perturbed_df['Frequency_original'].dropna()))[k - 1]

    # For violations of soundness, find the original frequencies < f_k - y
    soundness_candidates = original_vs_perturbed_df[original_vs_perturbed_df['Frequency_original'] < f_k - y]
    # Drop any bins that were not returned in the top k output
    returned_soundness_candidates = soundness_candidates.dropna()
    # The number of records in this dataframe is the number of rows that had original frequency < f_k - y but were still returned
    violations_of_soundness = len(returned_soundness_candidates)

    # For violations of completeness, count how many original frequencies > f_k + y are present
    completeness_candidates = original_vs_perturbed_df[original_vs_perturbed_df['Frequency_original'] > f_k + y]
    # Then drop any bins that were not returned in the top k output
    returned_completeness_candidates = completeness_candidates.dropna()
    # The difference of these two dataframes is the number of candidates that had freq > f_k + y and were not returned
    violations_of_completeness = len(completeness_candidates) - len(returned_completeness_candidates)

    # For false positives, find the original samples that had frequency < f_k
    true_exclude_top_k_df = original_vs_perturbed_df[original_vs_perturbed_df['Frequency_original'] < f_k]
    # The false positives will be the number of records in this dataframe that were returned
    false_positives = len(true_exclude_top_k_df.dropna())

    # For false negatives, first find the true top k
    true_top_k_df = original_vs_perturbed_df[original_vs_perturbed_df['Frequency_original'] >= f_k]
    # The missing candidates will be the elements of this dataframe that were not returned
    false_negatives = len(true_top_k_df) - len(true_top_k_df.dropna())

    # For violations of frequency range, count the occurrences where the frequencies differ by eta
    frequency_range_variation = np.abs(
        original_vs_perturbed_df['Frequency'] - original_vs_perturbed_df['Frequency_original'])
    violations_of_frequency_range = len(frequency_range_variation[frequency_range_variation > eta])

    # Save outputs
    outputs = {
        'violations_of_completeness': violations_of_completeness,
        'violations_of_soundness': violations_of_soundness,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'violations_of_frequency_range': violations_of_frequency_range,
        'f_k': float(f_k),
        'y': float(y),
        'eta': float(eta),
        'epsilon_1': epsilon_1,
        'epsilon_2': epsilon_2,
        'previous_revision_epsilon': previous_epsilon,
    }
    return outputs
