"""
Goal of Task 1:
    Prepare dataset to later train an object detection network.
"""


import os
from os import listdir


def create_train_txt(dataset_path):
    """
    Function that creates a file named "train.txt" at the highest level of your dataset folder.
    Example content for train.txt (also see train_example.txt in kitti folder):
        <
            000000.png
            000001.png
            ...
            999999.png
        >

    input:
        dataset_path (type: str): RELATIVE file path to the highest level of the dataset (e.g. kitti)

    output:
        img_names (type: list): list with all image names contained in task1/kitti/images/train
    """

    # Task:
    # ToDo: Write code to create the file "train.txt" as outlined in the task description on codefreak. In addition,
    #   this function should output the image names in a list (without the empty last line).
    # Hints:
    #   - Use the already imported os function "listdir" to list all filenames in a directory and the function "open()"
    #     to create and write into the file.
    #   - The length of the returned img_names list should be equal to the number of images in task1/kitti/images/train.
    #   - The evaluation on codefreak will be conducted with a different dataset, make sure your code is generic!
    #   - The last line of the created train.txt file should be an empty line.
    ########################
    #  Start of your code  #
    #######################
    data_dir = dataset_path
    img_dir = os.path.join(data_dir, "images/train")
    img_names = list()
    with open(os.path.join(os.path.dirname(os.path.dirname(dataset_path)) + 'train.txt'), 'w') as f:
        for file in os.listdir(img_dir):
            if file.endswith(".png"):
                f.write("%s\n" % file)
                img_names.append(file)
    ########################
    #   End of your code   #
    ########################

    return img_names


if __name__ == "__main__":
    dataset_path = "kitti"
    image_names = create_train_txt(dataset_path)
    print(f"Images: {image_names}")
    print(f"Number of images: {len(image_names)}")
