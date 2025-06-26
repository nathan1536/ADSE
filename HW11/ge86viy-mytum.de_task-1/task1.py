# coding:utf-8

"""
Goal of Task 1:
    Get an overview of how the data set can be prepared, which fulfills the Markov Property.

Hint:
    The dataset is a modified and very small version of http://rpg.ifi.uzh.ch/RAMNet.html.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
from PIL import Image


module_path = os.path.dirname(os.path.abspath(__file__))


class MarkovProcess(torch.utils.data.Dataset):
    def __init__(self, dir_name, transform=None):
        """
        inputs:
            dir_name (type: str): data set path
            transform: transform of the state image
        """

        self.images = []
        steering = []
        throttle = []
        brake = []

        # Subtask 1:
        # ToDo: Extract the required state and action information from the data set. Since there are too many images
        #  to load all of them into the memory, only the paths to the respective RGB images should be stored in the
        #  corresponding variable. The necessary control, throttle and brake commands should be read from the txt file
        #  with the help of pandas.read_csv and saved directly into the *_raw variables from the resampling code.
        # Hints:
        #   - You don't need to think about resampling due to different sampling rates of image and vehicle data. The
        #     "resampling code" is already given (see Subtask 2).
        #   - Your solution should work for a variable amount of towns and sequences.
        #   - Use the os package for extracting the paths.
        ########################
        #  Start of your code  #
        ########################
        lists = pd.read_csv(dir_name)["file"].values.tolist()
        for d_ in lists:
            df = pd.DataFrame(pd.read_pickle(d_))
            s_ = df.iloc[:, 0].values.tolist()
            a_ = df.iloc[:, 1].values.tolist()
            self.state_old.extend(s_[:-1])
            self.action.extend(a_[:-1])

        ########################
        #   End of your code   #
        ########################

        # Subtask 2:
        # ToDo: Uncomment these lines ("resampling code").
        # # due to a higher sampling rate of vehicle action data we need to resample the actions and append
        # # only every 40th measurement point
        for idx, (steering_, throttle_, brake_) in enumerate(zip(steering_raw, throttle_raw, brake_raw)):
            if idx % 40 == 0:
                steering.append(steering_)
                throttle.append(throttle_)
                brake.append(brake_)

        self.action = [steering, throttle, brake]
        self.action = np.array(self.action).T
        self.n_data_action = self.action.shape[0]
        self.n_data_state = len(self.images)
        self.dtype = np.uint8
        self.transform = transform

        print("number of data: {}".format(self.n_data_action))

    def __len__(self):
        return self.n_data_action

    def __getitem__(self, idx):
        """
        input:
            idx (type: int): index of the specific state-action pair to be returned

        outputs:
            state: type and shape can depend on transformation!
            action (shape: (3,)): action including the steering, throttle and brake command
        """

        # Subtask 3:
        # ToDo: Return the corresponding state-action pair.
        # Hints:
        #   - Be aware about a possible transformation of the state. It is stored in the variable self.transform.
        #   - You don't need to normalize the state and actions.
        ########################
        #  Start of your code  #
        ########################
        if idx < self.n_data_state:
            state = self.images[idx].astype(self.dtype)
            action = self.action[idx].astype(np.float32)
            if self.transform is not None:
                state = self.transform(state)
        ########################
        #   End of your code   #
        ########################

        return state, action


if __name__ == '__main__':
    print("test dataset and its dataloader")
    LDIR = "expert"
    BATCH_SIZE = 1

    transformation = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 256)), transforms.ToTensor()])
    dataset = torch.utils.data.DataLoader(MarkovProcess(LDIR, transform=transformation), batch_size=BATCH_SIZE,
                                          shuffle=False)

    counter = 0
    for batch_idx, (so, a_) in enumerate(dataset):
        print(batch_idx, so.size(), a_.size())
        counter = counter + 1
        if counter % 10 == 0:
            plt.imshow(so[0].permute(1, 2, 0))
            plt.show()
