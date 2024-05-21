import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stgcn_data import Feeder
from stgcn_model import ST_GCN
from matplotlib import pyplot as plt
import json
import os
import pickle

seed = 123
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

NUM_EPOCH = 30
BATCH_SIZE = 64

# モデルを作成
model = ST_GCN(num_classes=6, 
                  in_channels=3,
                  t_kernel_size=9, # 時間グラフ畳み込みのカーネルサイズ (t_kernel_size × 1)
                  hop_size=2).cuda()

# オプティマイザ
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 誤差関数
criterion = torch.nn.CrossEntropyLoss()

# データセットの用意
data_loader = dict()
data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(data_path='train_data.npy', label_path='train_labels.npy'), batch_size=BATCH_SIZE, shuffle=True,)
data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(data_path='test_data.npy', label_path='test_labels.npy'), batch_size=BATCH_SIZE, shuffle=False)

# モデルを学習モードに変更
model.train()

# 学習開始
for epoch in range(1, NUM_EPOCH+1):
  correct = 0
  sum_loss = 0
  for batch_idx, (data, label) in enumerate(data_loader['train']):
    data = data.cuda()
    label = label.cuda()

    output = model(data)

    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
    _, predict = torch.max(output.data, 1)
    correct += (predict == label).sum().item()

  print('# Epoch: {} | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, sum_loss/len(data_loader['train'].dataset), (100. * correct / len(data_loader['train'].dataset))))


# モデルを評価モードに変更
model.eval()

correct = 0
confusion_matrix = np.zeros((6, 6))
with torch.no_grad():
  for batch_idx, (data, label) in enumerate(data_loader['test']):
    data = data.cuda()
    label = label.cuda()

    output = model(data)
    
    _, predict = torch.max(output.data, 1)
    correct += (predict == label).sum().item()

    for l, p in zip(label.view(-1), predict.view(-1)):
      confusion_matrix[l.long(), p.long()] += 1

len_cm = len(confusion_matrix)
for i in range(len_cm):
    sum_cm = np.sum(confusion_matrix[i])
    for j in range(len_cm):
        confusion_matrix[i][j] = 100 * (confusion_matrix[i][j] / sum_cm)

classes = ['Cello', 'Violin', 'Clarinet', 'Flute', 'Horn', 'Trombone']
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.tight_layout()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('graph.jpg')

instrument_dirs = [
        '../solos_data/keypoints_data/Cello',
        '../solos_data/keypoints_data/Violin',
        '../solos_data/keypoints_data/Clarinet',
        '../solos_data/keypoints_data/Flute',
        '../solos_data/keypoints_data/Horn',
        '../solos_data/keypoints_data/Trombone'
    ]
instruments =['Cello', 'Violin', 'Clarinet', 'Flute', 'Horn', 'Trombone']

with torch.no_grad():
  for instrument, inst in zip(instrument_dirs, instruments):
        if(not os.path.exists(os.path.join("../data/solos/features",inst, "feature_keypoint_dim64_21.5fps"))):
               os.mkdir(os.path.join("../data/solos/features",inst, "feature_keypoint_dim64_21.5fps"))
        base_dir = instrument
        file_pattern = 'video_{:05d}.npy'
        picklefile_pattern = 'video_{:05d}.pkl'
        for i in range(1000):
          data_loader['generate'] = torch.utils.data.DataLoader(dataset=Feeder(os.path.join(base_dir,file_pattern.format(i)), label_path='temp_labels.npy'), batch_size=215, shuffle=False,)
          for batch_idx, (data, label) in enumerate(data_loader['generate']):
            data = data[:214].cuda()
            output = model.get_feature(data)
            ### pickleで保存（書き出し）
            with open(os.path.join("../data/solos/features",inst, "feature_keypoint_dim64_21.5fps",picklefile_pattern.format(i)), mode='wb') as fo:
              pickle.dump(output, fo)
            
         
          
