import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def add_images_path(input_data):
    """
    This function adds new column with image paths for each data.

    :param input_data: dataframe to be processed
    :return: input_data - dataframe with ned column with image path
    """

    image_part_1_path = '../data/input/HAM10000_images_part_1/'
    image_part_2_path = '../data/input/HAM10000_images_part_2/'
    extension = '.jpg'

    input_data['image_path'] = input_data.apply(lambda x: np.where(
        os.path.exists(str(image_part_1_path + x['image_id'] + extension)),
        str(image_part_1_path + x['image_id'] + extension),
        str(image_part_2_path + x['image_id'] + extension)),
                                                axis=1)

    file_path = str(input_data.loc[0]['image_path'])
    plt.title("Disease Image")
    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixels scaling")

    image = mpimg.imread(file_path)
    plt.imshow(image)
    plt.show()

    return input_data


def add_disease_classes(added_images):
    """
    This function adds new column which contains codes of each disease.

    :param added_images: input dataframe to be processed
    :return: added_images - dataframe with new column with diseases codes
    """

    diseases_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    added_images['diseases'] = added_images.apply(lambda x: diseases_dict[x['dx']],
                                                  axis=1)

    added_images['diseases_code'] = pd.Categorical(added_images['diseases']).codes

    return added_images


def main():
    input_data = pd.read_csv('../data/preprocessing/dropped_unknown_sex_localization_nulls.csv',
                             header=0,
                             index_col=0)

    added_images_path = add_images_path(input_data)
    added_diseases_classes = add_disease_classes(added_images_path)

    added_diseases_classes.to_csv(
        r"C:\Users\stank\Desktop\Faks\Master\Master rad\Skin pattern recognition using "
        r"ML\skin_patterns_recognition_using_ML\data\preprocessing\added_images_diseases_classes.csv")


if __name__ == '__main__':
    main()
