import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_analysis(input_data):
    """
    This function creates analysis of input data.

    :param input_data: dataframe to be processed
    :return: None
    """

    print(f"Columns are: {input_data.columns}")

    dx_unique = pd.unique(input_data['dx'])
    print(f"Unique dx: {dx_unique}")

    dx_type_unique = pd.unique(input_data['dx_type'])
    print(f"Unique dx_type: {dx_type_unique}")

    age_unique = pd.unique(input_data['age'])
    print(f"Unique age: {len(age_unique)}")

    sex_unique = pd.unique(input_data['sex'])
    print(f"Unique sex: {sex_unique}")

    localization_unique = pd.unique(input_data['localization'])
    print(f"Unique localization: {localization_unique}")

    id_unknown_sex = pd.unique(input_data[input_data['sex'] == 'unknown']['image_id'])
    id_unknown_localization = pd.unique(input_data[input_data['localization'] == 'unknown']['image_id'])
    print(f"Unknown sex: {len(input_data[input_data['sex'] == 'unknown'])}")
    print(f"Unknown sex ids: {id_unknown_sex}")

    print(f"Unknown localization: {len(input_data[input_data['localization'] == 'unknown'])}")
    print(f"Unknown localization ids: {id_unknown_localization}")

    same_unknown_ids = np.intersect1d(id_unknown_sex, id_unknown_localization)
    print(f"Same unknown ids: {len(same_unknown_ids)}")

    print(f"Analysis:\n{input_data.isnull().sum()}")


def data_visualization(input_data):
    """
    This function visualize input data so it could be analysed.

    :param input_data: dataframe which contains data to be visualized
    :return: None
    """

    my_colors = ['black', 'red', 'green', 'blue', 'cyan', 'silver', 'gold', 'slategrey', 'crimson', 'olive', 'orange',
                 'tomato', 'navy', 'lime', 'violet']

    input_data['dx'].value_counts().plot(kind='bar', color=my_colors)

    x = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions', 'Basal cell carcinoma', 'Actinic keratoses',
         'Vascular lesions', 'Dermatofibroma']
    values = np.arange(0, 7, 1)

    plt.xticks(values, x,
               rotation=90)
    plt.title("Diseases")
    plt.xlabel("Diseases type")
    plt.ylabel("Count")
    plt.savefig('input_data_analysis_results/diseases_type_graph.png',
                bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    input_data['dx_type'].value_counts().plot(kind='bar',
                                              color=my_colors)

    plt.title("Technical validation")
    plt.xlabel("Technical validation type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('input_data_analysis_results/technical_validation_graph.png',
                bbox_inches='tight')

    plt.show()

    input_data['age'].hist(color='darkred',
                           histtype='bar',
                           ec='black')

    plt.title("Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig('input_data_analysis_results/age_graph.png',
                bbox_inches='tight')

    plt.show()

    input_data['sex'].value_counts().plot(kind='pie')
    plt.title("Sex")
    plt.legend()
    plt.tight_layout()
    plt.savefig('input_data_analysis_results/sex_graph.png',
                bbox_inches='tight')

    plt.show()

    input_data['localization'].value_counts().plot(kind='bar',
                                                   color=my_colors)

    plt.title("Localization")
    plt.xlabel("Localization place")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('input_data_analysis_results/localization_graph.png',
                bbox_inches='tight')

    plt.show()

    plt.title("Diseases depending on age")
    plt.xlabel("Disease")
    plt.ylabel("Age")
    plt.xticks(values, x,
               rotation=90)

    plt.scatter(input_data['dx'], input_data['age'],
                color="red")

    plt.tight_layout()
    plt.savefig('input_data_analysis_results/diseases_depending_on_age.png',
                bbox_inches='tight')

    plt.show()

    plt.title("Diseases depending on localization")
    plt.xlabel("Disease")
    plt.ylabel("Sex")
    plt.xticks(values, x,
               rotation=90)

    plt.scatter(input_data['dx'], input_data['sex'],
                color="blue")

    plt.tight_layout()
    plt.savefig('input_data_analysis_results/diseases_depending_on_sex.png',
                bbox_inches='tight')

    plt.show()

    plt.title("Diseases depending on localization")
    plt.xlabel("Disease")
    plt.ylabel("Localization")
    plt.xticks(values, x,
               rotation=90)

    plt.scatter(input_data['dx'], input_data['localization'],
                color="green")

    plt.tight_layout()
    plt.savefig('input_data_analysis_results/diseases_depending_on_localization.png',
                bbox_inches='tight')

    plt.show()

    plt.title("Localization depending on sex")
    plt.xlabel("Localization")
    plt.ylabel("Sex")
    plt.xticks(rotation=90)
    plt.scatter(input_data['localization'], input_data['sex'],
                color="gray")

    plt.tight_layout()
    plt.savefig('input_data_analysis_results/localization_depending_on_sex.png',
                bbox_inches='tight')

    plt.show()


def main():
    input_data = pd.read_csv('../data/input/input_data_from_kaggle.csv',
                             header=0,
                             index_col=0)

    data_analysis(input_data)

    data_visualization(input_data)


if __name__ == '__main__':
    main()
