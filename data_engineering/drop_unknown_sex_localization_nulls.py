import pandas as pd
import numpy as np


def drop_unknown_sex_localization(input_data):
    """
    This function drops data which has unknown sex or localization cell.

    :param input_data: dataframe to be processed
    :return: input_data - input dataframe with dropped unknown sex or localization cell
    """

    ids_unknown_sex = pd.unique(input_data[input_data['sex'] == 'unknown']['image_id'])
    ids_unknown_localization = pd.unique(input_data[input_data['localization'] == 'unknown']['image_id'])
    ids_unknown = np.unique(np.append(ids_unknown_sex, ids_unknown_localization))

    input_data.drop(index=input_data.loc[input_data['image_id'].isin(ids_unknown)].index,
                    axis=0,
                    inplace=True)

    input_data = input_data.reset_index(drop=True)

    return input_data


def drop_missing_data(dropped_unknown_sex_localization):
    """
    This function drops data which has any null cells.

    :param dropped_unknown_sex_localization: dataframe to be processed
    :return: dropped_unknown_sex_localization - input dataframe with dropped null cells
    """

    print(f"Number of data which has nulls: {dropped_unknown_sex_localization['age'].isna().sum()}")
    dropped_unknown_sex_localization.drop(
        index=dropped_unknown_sex_localization.loc[dropped_unknown_sex_localization['age'].isna()].index,
        axis=0,
        inplace=True)

    dropped_unknown_sex_localization = dropped_unknown_sex_localization.reset_index(drop=True)

    return dropped_unknown_sex_localization


def main():
    input_data = pd.read_csv('../data/input/input_data_from_kaggle.csv',
                             header=0,
                             index_col=0)

    dropped_unknown_sex_localization = drop_unknown_sex_localization(input_data)

    dropped_missing_data = drop_missing_data(dropped_unknown_sex_localization)

    dropped_missing_data.to_csv(
        r"C:\Users\stank\Desktop\Faks\Master\Master rad\Skin pattern recognition using "
        r"ML\skin_patterns_recognition_using_ML\data\preprocessing\dropped_unknown_sex_localization_nulls.csv")


if __name__ == '__main__':
    main()
