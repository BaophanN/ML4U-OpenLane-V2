import warnings 
import sys
sys.path.insert(0, '/workspace/source')
from openlanev2.lanesegment.io import io
from openlanev2.lanesegment.preprocessing import collect

root_path = '/workspace/source/data/datasets'
file = f'{root_path}/data_dict_subset_B.json'
subset = 'data_dict_subset_B'

from openlanev2.lanesegment.dataset import Collection, Frame

collection = Collection(root_path, root_path, 'data_dict_sample_ls')
frame = collection.get_frame_via_identifier(('train', '00492', '315970276749927222'))

for k, v in frame.get_pose().items():
    # each frame - contains many views - has a common pose
    print(k, '\n', v, '\n')
for k in frame.get_camera_list():
    print(k)

camera = frame.get_camera_list()[0]
'''
{'rotation': array([[-0.03161852,  0.99932457,  0.01872611],
       [-0.99904989, -0.03103652, -0.0305949 ],
       [-0.02999304, -0.01967568,  0.99935644]]), 'translation': array([6657.19639943, 1842.86566032,   59.89421563])}
'''
meta = {
    # each camera has their own parameters 
    'intrinsic': frame.get_intrinsic(camera),
    'extrinsic': frame.get_extrinsic(camera),
}
for key, value in meta.items():
    for k, v in value.items():
        print(key, '-', k, '\n', v, '\n')
for key in frame.__dict__.keys():
    print(key)
'''
intrinsic - K 
 [[1.77753967e+03 0.00000000e+00 7.77762878e+02]
 [0.00000000e+00 1.77753967e+03 1.01631311e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

intrinsic - distortion 
 [-0.24479243 -0.19577468  0.30131747] 

extrinsic - rotation 
 [[-8.48589870e-04  1.00005773e-02  9.99949633e-01]
 [-9.99998983e-01 -1.15429954e-03 -8.37087508e-04]
 [ 1.14587004e-03 -9.99949327e-01  1.00015467e-02]] 

extrinsic - translation 
 [1.63315125 0.00800013 1.38385219] 
'''
"""
root_path 
meta 
"""
print('-----')
for key in frame.meta: 
    print(key)
print('-----')
"""
version
segment_id
meta_data
timestamp
sensor
annotation
pose
"""
for key in frame.meta['meta_data']['source']:
    # 'meta_data'
    
    print(frame.meta['meta_data']['source_id'])

   

# print(frame.meta['sensor'][camera]['intrinsic'])
print(frame.meta['pose'])
print("-----")
print(frame.meta['sensor'])

# cam_name, cam_info = frame.meta['sensor']


    
