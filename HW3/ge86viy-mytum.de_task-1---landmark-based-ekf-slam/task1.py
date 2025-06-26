"""
Goal of Task 1:
    Implement an automated map-generator.

Hint: Use the already implemented EKF SLAM in EKF_SLAM.py to test your code.
"""

import random


def GenerateLandmarks(x_min, x_max, y_min, y_max, n):
    """
    inputs:
        x_min (type: int): lower limit of x-coordinate
        x_max (type: int): upper limit of x-coordinate
        y_min (type: int): lower limit of y-coordinate
        y_max (type: int): upper limit of y-coordinate
        n (type: int): number of landmarks to be generated

    output:
        landmarks (type: np.ndarray, shape (n,2)): [x, y] - points for all n landmarks
    """
    # Task:
    # ToDo: Generate n randomly positioned landmarks within the given range.
    ########################
    #  Start of your code  #
    ########################
    i = 0
    landmarks = [0] * n
    while i < n:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        landmarks[i] = [x, y]
        i = i + 1
    print(landmarks)
    ########################
    #   End of your code   #
    ########################
    return landmarks

GenerateLandmarks(10, 30, 20, 40, 5)

