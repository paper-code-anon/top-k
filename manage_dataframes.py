import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_dataframe(dataframe_name, filename, week=None, program_id=None, attribute=None):
    if dataframe_name == 'scores':
        return get_scores_dataframe(filename, week, program_id)
    elif dataframe_name == 'activities':
        return get_activities_dataframe(filename, week, program_id)
    elif dataframe_name == 'fake_scores':
        return get_fake_scores_dataframe(filename, week)
    elif dataframe_name == 'kaggle_user_activity':
        return get_kaggle_user_activity_dataframe(filename)
    elif dataframe_name == 'hospital':
        return get_hospital_dataframe(filename, attribute)
    elif dataframe_name == 'dummy':
        return get_dummy_data(filename)
    else:
        raise ValueError('Error: Invalid dataframe_name')


def get_hospital_dataframe(filename, attribute):
    df = pd.read_csv(filename)
    if attribute not in df.columns:
        raise ValueError('Hospital attribute not found')
    s = df[attribute]

    return s.value_counts()


def get_activities_dataframe(filename, week=None, program_id=None):
    df = pd.read_csv(filename)
    if week:
        df = df[df['week'] == week]
    if program_id:
        df = df[df['program_id'] == program_id]
    # Process the date to split into hour, and time interval [0-30, 30-60]
    df['hour'] = df['time'].map(lambda x: x.split(':')[0])
    df['min'] = df['time'].map(lambda x: '0-30' if int(x.split(':')[1]) < 30 else '30-60')
    df['datetime'] = df.date.astype(str).str.cat(df['hour'].astype(str), sep=' ').str.cat(df['min'].astype(str), sep=':')
    # Select only the relevant attributes and drop any duplicates to find number of unique students interacting
    # in a given timeslot
    df = df[['student_id', 'datetime']].drop_duplicates()

    # Group by date and time to get frequencies
    # s = df['datetime'].value_counts(sort=True, ascending=False)
    s = df['datetime'].value_counts()
    # Convert index to datetime index
    s.index = s.index.map(lambda x: x[0:-3])
    s.index = pd.DatetimeIndex(s.index)

    # Get date range based on index values
    min_date = min(s.index).round(freq='D')
    max_date = max(s.index).round(freq='D')
    # Generate range for program and cut off last date
    program_range = pd.date_range(start=min_date, end=max_date, freq='30min')[0:-1]
    s = s.reindex(program_range, fill_value=0)
    # Format to blocks of 30
    s.index = s.index.format()
    s.index = s.index.map(lambda x: x[0:-5] + '00-30' if x.endswith('00:00') else x[0:-5] + '30-60')
    # Sort
    s = s.sort_values(ascending=False)
    return s


def get_scores_dataframe(filename, week=None, program_id=None):
    df = pd.read_csv(filename)
    if week:
        df = df[df['week'] == week]
    if program_id:
        df = df[df['program_id'] == program_id]
    df_binned = get_histogram(df, 'moderated_score.max')
    # Remove categorical index as we need to modify it later
    df_binned.index = df_binned.index.tolist()
    return df_binned


def get_histogram(df, col_name):
    return pd.cut(df[col_name],
                  [x * 0.1 for x in range(0, 11, 1)],
                  include_lowest=True,
                  labels=[f'{round((x - 1) * 0.1, 1)}-{round(x * 0.1, 1)}' for x in range(1, 11, 1)])\
        .value_counts(sort=True, ascending=False)


def get_fake_scores_dataframe(filename, week=None):
    df = pd.read_csv(filename)[['team_id', 'week1', 'week2', 'week3']].drop_duplicates(subset='team_id', keep='last')
    if week:
        df_binned = get_histogram(df, week)
    else:
        df = pd.DataFrame(pd.concat([df['week1'], df['week2'], df['week3']]), columns=['all_weeks'])
        df_binned = get_histogram(df, 'all_weeks')

    # Remove categorical index as we need to modify it later
    df_binned.index = df_binned.index.tolist()

    return df_binned


def get_kaggle_user_activity_dataframe(filename):
    df = pd.read_csv(filename)
    # Sum interactions to single parameter
    df['Frequency'] = df['datasets'] + df['submissions'] + df['scripts'] + df['comments']
    df = df.drop(['submissions', 'scripts', 'comments', 'datasets', 'username'], axis=1)
    # Group interactions across users
    df = df.groupby(by='date').sum()
    # Convert to pandas series for compatibility
    s = df['Frequency'].sort_values(ascending=False)

    # Get date range based on index values
    s.index = pd.DatetimeIndex(s.index)
    min_date = min(s.index).round(freq='D')
    max_date = max(s.index).round(freq='D')
    # Generate range for program and cut off last date
    program_range = pd.date_range(start=min_date, end=max_date, freq='D')
    s = s.reindex(program_range, fill_value=0)
    s.index = s.index.format()
    # Sort
    s = s.sort_values(ascending=False)

    return s

def get_dummy_data(filename):
    s = pd.read_csv(filename, index_col='Value')['Frequency']
    s = s.sort_values(ascending=False)
    return s


def save_practera_results(path_name, df_pert, df_original, title_original, title_perturbed, x_axis_name, week_name=None):
    plt.figure(1, figsize=(23, 10))
    plt.subplot(211)
    plt.subplots_adjust(hspace=0.6)
    df_top_10 = df_original.iloc[0:10]
    plt.bar(np.arange(len(df_top_10)), df_top_10.values)
    plt.xticks(np.arange(len(df_top_10)),
               map(lambda x: x if type(x) is str else round(x, 2), list(df_top_10.index.values)))
    plt.title(title_original)
    plt.xlabel(x_axis_name)
    plt.ylabel('Frequency')

    plt.subplot(212)
    plt.bar(np.arange(len(df_pert)), df_pert.values)
    plt.xticks(np.arange(len(df_pert)), map(lambda x: x if type(x) is str else round(x, 2), list(df_pert.index.values)))
    plt.title(title_perturbed)
    plt.xlabel(x_axis_name)
    plt.ylabel('Perturbed Frequency')
    # plt.show()
    plt.savefig(f'{path_name}/{week_name if week_name else "all_3_weeks"}.png')
    df_pert.to_csv(f'{path_name}/{week_name if week_name else "all_3_weeks"}_top_k.csv', header=['Frequency'], index_label=[x_axis_name])
    df_original.columns = [x_axis_name, 'Frequency']
    df_original.to_csv(f'{path_name}/{week_name if week_name else "all_3_weeks"}_original_data.csv', header=['Frequency'], index_label=[x_axis_name])
    plt.clf()


def _save_general_outputs(path_name, df_pert, df_original, title_perturbed, title, xlabel):
    plt.figure(1, figsize=(23, 12))
    plt.subplot(211)
    plt.subplots_adjust(hspace=0.8)
    df_top_10 = df_original.iloc[0:10]
    plt.bar(np.arange(len(df_top_10)), df_top_10.values)
    plt.xticks(np.arange(len(df_top_10)),
               map(lambda x: x if type(x) is str else round(x, 2), list(df_top_10.index.values)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xticks(rotation=15)


    plt.subplot(212)
    plt.bar(np.arange(len(df_pert)), df_pert.values)
    plt.xticks(np.arange(len(df_pert)), map(lambda x: x if type(x) is str else round(x, 2), list(df_pert.index.values)))
    plt.title(title_perturbed)
    plt.xlabel(xlabel)
    plt.ylabel('Perturbed Frequency')
    # plt.show()
    plt.xticks(rotation=15)


    plt.savefig(f'{path_name}/pert_top_k.png')
    df_pert.to_csv(f'{path_name}/pert_top_k.csv', header=['Frequency'], index_label=[xlabel])
    df_original.columns = [xlabel, 'Frequency']
    df_original.to_csv(f'{path_name}/original_data.csv', header=['Frequency'], index_label=[xlabel])
    plt.clf()


def save_general_outputs(dataframe_name, path_name, df_pert, df_original, title_perturbed):
    if dataframe_name == 'kaggle_user_activity':
        _save_general_outputs(path_name, df_pert, df_original, title_perturbed,
                              'Top 10 true values from kaggle dataset', 'Date')
    elif dataframe_name == 'hospital':
        _save_general_outputs(path_name, df_pert, df_original, title_perturbed,
                              'Top 10 true values from hospital dataset', 'Value')
    elif dataframe_name == 'dummy':
        _save_general_outputs(path_name, df_pert, df_original, title_perturbed,
                              'Top 10 true values from dummy dataset', 'Value')
    else:
        raise ValueError('Invalid dataframe name')


