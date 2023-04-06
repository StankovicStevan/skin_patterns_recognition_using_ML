import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_dummies_categorical_columns(input_data):
    """
    This function creates dummy columns from categorical column, exactly "dx_type" and "localization" columns.

    :param input_data: dataframe to be processed
    :return: input_data - dataframe with dummy columns
    """

    input_data = pd.get_dummies(input_data,
                                columns=["dx_type", "localization"])

    return input_data


def label_encoding_sex_column(df_with_dummies):
    """
    This function converts categorical sex column to numerical one.

    :param df_with_dummies: dataframe to be processed
    :return: df_with_dummies - dataframe with converted column
    """

    le = LabelEncoder()
    df_with_dummies['sex'] = le.fit_transform(df_with_dummies['sex'])

    return df_with_dummies


def main():
    input_data = pd.read_csv('../data/preprocessing/added_images_diseases_classes.csv',
                             header=0,
                             index_col=0)

    df_with_dummies = create_dummies_categorical_columns(input_data)

    df_with_encoded_sex_columns = label_encoding_sex_column(df_with_dummies)

    df_with_encoded_sex_columns.to_csv(
        r"C:\Users\stank\Desktop\Faks\Master\Master rad\Skin pattern recognition using "
        r"ML\skin_patterns_recognition_using_ML\data\preprocessing\converted_categorical_columns.csv")


if __name__ == '__main__':
    main()
