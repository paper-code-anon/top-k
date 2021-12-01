import pandas as pd
import numpy as np
from top_k_algorithms import top_k_algorithm_1
from manage_dataframes import get_dataframe, save_general_outputs
from scipy.stats import logistic
import matplotlib.pyplot as plt



def calculate_gamma_for_violations_of_completeness(df_counts, beta, k, k_bar, epsilon_1, little_delta, little_delta_p,
                                    big_delta):
    f_k = df_counts.iloc[k - 1]
    f_kbar_plus_1 = df_counts.iloc[k_bar]
    print(f'f_k is: {f_k}')
    print(f'f_kbar_plus_1 is: {f_kbar_plus_1}')


    gamma_original = (1 / epsilon_1) * np.log(k * k_bar / beta)

    gamma_new = max((1 / epsilon_1) * np.log(k * k_bar / beta),
                    f_kbar_plus_1 + 1 + (1 / epsilon_1) * np.log(min(big_delta, k_bar) / little_delta) - f_k +
                    (1 / epsilon_1) * np.log((k - beta) / beta)
                    )
    print((1 / epsilon_1) * np.log(k * k_bar / beta))
    print(f_kbar_plus_1 + 1 + (1 / epsilon_1) * np.log(min(big_delta, k_bar) / little_delta) - f_k)
    print((1 / epsilon_1) * np.log((k - beta) / beta))
    return gamma_original, gamma_new, df_counts > f_k + gamma_original, df_counts > f_k + gamma_new,


def check_if_completeness_was_achieved(df_original, df_perturbed, f_k, gamma):

    # Our completeness candidates are all original values with f_i > f_k + gamma
    completeness_check = df_original[df_original > f_k + gamma]
    completeness_candidates = list(completeness_check.index)
    # We achieve completeness if all completeness candidates are in the perturbed histogram
    completeness_achieved = all(candidate in df_perturbed.index for candidate in completeness_candidates)
    return completeness_achieved



if __name__ == '__main__':
    number_of_iterations_to_average = 100  # Sets how many iterations we want to calculate analytics for
    beta_range = [x * 0.01 for x in range(1, 100)]

    """Input Parameters"""
    # Set privacy input Parameters
    # Produced second figure
    filename = 'test_break_new_gamma.csv'
    little_delta = 0.001
    little_delta_prime = 0.001
    big_delta = 1
    k = 3
    k_bar = 3
    epsilon = 26.5


    # Use helper function get_dataframe to correctly generate the histogram for either scores or activities
    df_counts = get_dataframe(dataframe_name='dummy',
                              filename=filename)
    f_k = df_counts.iloc[k - 1]

    _, _, epsilon_1, _ = top_k_algorithm_1(df_counts, epsilon, little_delta, little_delta_prime, big_delta, k, k_bar)

    original_completeness_rates = []
    new_completeness_rates = []
    original_gammas = []
    new_gammas = []


    for beta in beta_range:
        gamma_original, gamma_new, \
        exceeding_gamma_check_original, exceeding_gamma_check_new = calculate_gamma_for_violations_of_completeness(df_counts,
                                                                                                          beta,
                                                                                                          k,
                                                                                                          k_bar,
                                                                                                          epsilon_1,
                                                                                                          little_delta,
                                                                                                          little_delta_prime,
                                                                                                          big_delta,)
        original_gammas.append(gamma_original)
        new_gammas.append(gamma_new)

        # Apply top_k_algorithm_1 number_of_iterations_to_average times and store results
        perturbed_results = []
        original_completeness_achieved_results = []
        new_completeness_achieved_results = []
        for i in range(number_of_iterations_to_average):
            # Perturb the histogram and save it in our results array
            df_pert, v_perp, _, _ = top_k_algorithm_1(df_counts, epsilon, little_delta, little_delta_prime, big_delta, k, k_bar)
            df_pert.name = f'Perturbed_{i}'
            perturbed_results.append(df_pert)
            # Check whether completness was achieved
            original_completeness_achieved_results.append(check_if_completeness_was_achieved(df_counts, df_pert, f_k, gamma_original))
            new_completeness_achieved_results.append(check_if_completeness_was_achieved(df_counts, df_pert, f_k, gamma_new))

        original_completeness_rate = len(list(filter(lambda x: x, original_completeness_achieved_results))) / number_of_iterations_to_average
        new_completeness_rate = len(list(filter(lambda x: x, new_completeness_achieved_results))) / number_of_iterations_to_average

        print(f'beta is: {beta}')
        print(f'Dataset achieves NEW completeness with rate {round(new_completeness_rate, 4)} vs a predicted minimum completeness rate {round(1 - beta, 4)}')
        print(f'Dataset achieves ORIGINAL completeness with rate {round(original_completeness_rate, 4)} vs a predicted minimum completeness rate {round(1 - beta, 4)}')
        print()

        original_completeness_rates.append(original_completeness_rate)
        new_completeness_rates.append(new_completeness_rate)

    plt.figure(figsize=(8, 6))
    plot_figure_axes_fontsize = 16
    plot_figure_xticks_fontsize = 14
    plot_figure_yticks_fontsize = 14
    plot_figure_legend_fontsize = 14

    plt.rc('legend', fontsize=plot_figure_legend_fontsize)
    plt.rc('axes', titlesize=plot_figure_axes_fontsize)
    plt.rc('axes', labelsize=plot_figure_axes_fontsize)
    plt.rc('xtick', labelsize=plot_figure_xticks_fontsize)
    plt.rc('ytick', labelsize=plot_figure_yticks_fontsize)

    plt.plot(beta_range, original_completeness_rates, label='Original Gamma')
    plt.plot(beta_range, new_completeness_rates, label='New Gamma')
    plt.plot(beta_range, [1 - beta for beta in beta_range], 'g-', label='Expected Minimum')
    plt.legend()
    plt.xlabel('Beta')
    plt.ylabel('Completeness Rate')
    # plt.title('Completeness Rate for Varying Beta, using both Original and New Calculated Gamma')
    plt.savefig('violation_of_completeness.png')
    plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(beta_range, original_gammas, label='Original Gamma Value')
    # plt.plot(beta_range, new_gammas, label='New Gamma Value')
    # plt.legend()
    # plt.xlabel('Beta')
    # plt.ylabel('Gamma')
    # plt.title('Comparison of Original and New calculated Gamma Values')
    # plt.show()