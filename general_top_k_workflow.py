import os
import json
import numpy as np

from top_k_algorithms import top_k_algorithm_1, top_k_algorithm_2, calculate_metrics
from manage_dataframes import get_dataframe, save_general_outputs


if __name__ == '__main__':
    """Start inputs"""
    filename = 'USER_ACTIVITY.csv'
    dataframe_name = 'kaggle_user_activity'  # Set to 'kaggle_user_activity' or 'hospital' or 'dummy'
    hospital_attribute_name = None  # Set to a column in the hospital dataset, or None if using other dataset

    top_k_algorithm_type = 1  # Set algorithm type = 1 or = 2
    number_of_iterations_to_average = 10  # Sets how many iterations we want to calculate analytics for

    # Set privacy input Parameters
    little_delta = 0.0001
    little_delta_prime = 0.0001
    big_delta = 1919
    k = 10
    k_bar = 300  # Set = None for algorithm 2
    beta = 0.1
    epsilon_inputs = [0.1, 1, 4, 7, 10, 20]

    """End inputs"""


    # Use helper function get_dataframe to correctly generate the histogram for either scores or activities
    df_counts = get_dataframe(dataframe_name=dataframe_name,
                              filename=filename,
                              attribute=hospital_attribute_name)
    d = len(df_counts) - k  # TODO: Note this


    for epsilon in epsilon_inputs:
        path_name = f'top_k_{dataframe_name}_eps_{epsilon}'
        if not os.path.exists(path_name):
            os.mkdir(path_name)

        if len(df_counts) < k:
            print(f'k = {k} is greater than number of bins {len(df_counts)}, skipping...')
            continue
        if k_bar and len(df_counts) < k_bar:
            print(f'k = {k_bar} is greater than number of bins {len(df_counts)}, skipping...')
            continue

        # Apply top k algorithm based on input parameters
        if top_k_algorithm_type == 1:
            df_pert, v_perp, epsilon_1, epsilon_2 = top_k_algorithm_1(df_counts, epsilon, little_delta, little_delta_prime, big_delta, k,
                                                k_bar)
        elif top_k_algorithm_type == 2:
            df_pert, epsilon_1, epsilon_2 = top_k_algorithm_2(df_counts, epsilon, little_delta, k)
            v_perp = None

        # Save results
        save_general_outputs(dataframe_name, path_name, df_pert, df_counts,
                            f'Frequency histogram of perturbed top-k values using Algorithm {top_k_algorithm_type}\n'
                            f'epsilon_g = {epsilon}, epsilon_1 = {round(epsilon_1, 2)}, epsilon_2 = {round(epsilon_2, 2)}, k = {k}, little_delta = {little_delta}',
                            )

        # Get the analytics for this top k run and save to file
        analytic_outputs = calculate_metrics(df_counts,
                                             df_pert,
                                             top_k_algorithm_type,
                                             epsilon,
                                             epsilon_1,
                                             epsilon_2,
                                             little_delta_prime,
                                             beta,
                                             k,
                                             k_bar=k_bar,
                                             d=d)
        if v_perp:
            analytic_outputs['v_perp'] = v_perp
        # Round to easier decimal precision
        analytic_outputs = {key: round(float(value), 6) for key, value in analytic_outputs.items()}

        with open(path_name + f'/analytics.json', 'w') as outfile:
            json.dump(analytic_outputs, outfile, indent=4)

        # Also save all our input parameters to file
        input_parameters_dict = {
            'top_k_algorithm_type': top_k_algorithm_type,
            'epsilon': epsilon,
            'little_delta': little_delta,
            'little_delta_prime': little_delta_prime,
            'big_delta': big_delta,
            'k': k,
            'k_bar': int(k_bar),
            'beta': beta,
            'v_perp': v_perp
        }
        with open(path_name + f'/input_parameters.json', 'w') as outfile:
            json.dump(input_parameters_dict, outfile, indent=4)

        # Calculate average analytics over 10 runs and save result in folder
        averaged_analytics = {}
        standard_devs = {}

        for _ in range(number_of_iterations_to_average):
            if top_k_algorithm_type == 1:
                df_pert, v_perp, epsilon_1, epsilon_2 = top_k_algorithm_1(df_counts, epsilon, little_delta, little_delta_prime,
                                                    big_delta, k, k_bar)
            elif top_k_algorithm_type == 2:
                df_pert, epsilon_1, epsilon_2 = top_k_algorithm_2(df_counts, epsilon, little_delta, k)
                v_perp = None

            analytic_outputs = calculate_metrics(df_counts,
                                                 df_pert,
                                                 top_k_algorithm_type,
                                                 epsilon,
                                                 epsilon_1,
                                                 epsilon_2,
                                                 little_delta_prime,
                                                 beta,
                                                 k,
                                                 k_bar=int(k_bar),
                                                 d=d)
            if v_perp:
                analytic_outputs['v_perp'] = v_perp

            # Sum the averaged analytics that have been calculated into a new dictionary
            # Store running sum of squared for stddev calculation
            for key, value in analytic_outputs.items():
                if key in averaged_analytics.keys():
                    averaged_analytics[key] += value / number_of_iterations_to_average
                    standard_devs[key] += ((value ** 2) / number_of_iterations_to_average)
                else:
                    averaged_analytics[key] = value / number_of_iterations_to_average
                    standard_devs[key] = ((value ** 2) / number_of_iterations_to_average)

        # Calculate standard deviation
        for key, value in standard_devs.items():
            variance = standard_devs[key] - (averaged_analytics[key] ** 2)
            # Handle small floating point errors resulting in slightly negative values
            standard_devs[key] = 0 if variance < 0 else np.sqrt(variance)

        # Round to easier decimal precision
        averaged_analytics_rounded = {key: round(value, 6) for key, value in averaged_analytics.items()}
        standard_devs_rounded = {key: round(value, 6) for key, value in standard_devs.items()}

        # Save
        with open(path_name + f'/analytics_{number_of_iterations_to_average}_runs.json', 'w') as outfile:
            json.dump(averaged_analytics_rounded, outfile, indent=4)
        with open(path_name + f'/standard_devs_{number_of_iterations_to_average}_runs.json', 'w') as outfile:
            json.dump(standard_devs_rounded, outfile, indent=4)