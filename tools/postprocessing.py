import pickle
import numpy as np
from typing import Union
import copy

def euclid_dis(point1, point2):
    # return np.sum(np.square(point1 - point2))
    return np.linalg.norm(point1-point2)

def postprocessing_help(_input: dict, confidence_thresh = 0.3, distance_thresh = 3):
    '''
    for a single frame
    input: dict{
    'lane_results' : [[], [], []],
    'bbox_results' : ... ,
    'lclc_results' : ... ,
    ...
    }
    '''
    num_lane_centerline = len(_input['lane_results'][0])
    _input = copy.deepcopy(_input)
    for i in range(num_lane_centerline):
        if _input['lane_results'][1][i] < confidence_thresh:
            continue

        for j in range(i + 1, num_lane_centerline):
            if _input['lane_results'][1][j] < confidence_thresh:
                continue 

            start_i = _input['lane_results'][0][i][0:3]
            end_i = _input['lane_results'][0][i][-9:-6]
            start_j = _input['lane_results'][0][j][0:3]
            end_j = _input['lane_results'][0][j][-9:-6] # centerline 3, left 3, right 3 

            # if np.min(np.array([euclid_dis(start_i, start_j), euclid_dis(start_i, end_j),
            #           euclid_dis(end_i, start_j), euclid_dis(end_i, end_j)])) < distance_thresh:
            #     _input['lclc_results'][i][j] = _input['lclc_results'][j][i] = 1 
                # print('process lane assigned')
            if np.min(np.array([euclid_dis(start_i, end_j)])) < distance_thresh:
                _input['lsls_results'][j][i] = 1 
            if np.min(np.array([euclid_dis(start_j, end_i)])) < distance_thresh:
                _input['lsls_results'][i][j] = 1 
    return _input
    
def postprocessing(_input: Union[dict,str]):

    if isinstance(_input, str): #if input is a path to pickle file
        file = open(_input, 'rb')
        _input = pickle.load(file)
    
    for frame in range(len(_input)):
        _input[frame] = postprocessing_help(_input[frame])
    print("-->POSTPROCESS")
    return _input

if __name__ == "__main__":
    
    print(postprocessing('./results.pkl'))
