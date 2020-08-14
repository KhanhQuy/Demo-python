import lightgbm as lgb
import numpy as np


train_data = lgb.Dataset('train.svm.bin')

data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(data, label=label)

print(train_data)

# train_data = lgb.Dataset('train.svm.txt')
# train_data.save_binary('train.bin')
