import numpy as np
FEATURES:int = 40


"""
    Given an input set of encoded chars, it will extend the input to the number of features in the configuration
"""
def extend_to_feature_length(input:np.array):

    output = np.zeros(FEATURES)
    output[0: len(input)] = input
    return output