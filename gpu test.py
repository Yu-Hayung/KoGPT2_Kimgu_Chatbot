
import torch
print('torch >>' ,torch.__version__)
print(torch.cuda.is_available())

# torch _ gpu 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print('+++++++'*20)

import os, tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib

print('tensorflow >>' ,tensorflow.__version__)
print('tensorflow :: \n', device_lib.list_local_devices())

'''

GPU:n번을 사용하려면 번호를 n으로 지정해주시면 되며,
(위의 예시에서는 GPU:0번이 사용됩니다.)
CPU 강제 사용을 원하신다면 -1로 번호를 선택해주시면 됩니다.

'''


import torch
print("torch > ", torch.FloatTensor(10))
