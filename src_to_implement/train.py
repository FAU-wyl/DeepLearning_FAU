import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
csv_path = 'data.csv' # Ensure this path is correct
df = pd.read_csv(csv_path, sep=';') # Check if your CSV uses ',' or ';'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_ds = ChallengeDataset(train_df, mode='train')
val_ds = ChallengeDataset(val_df, mode='val')

train_dl = t.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = t.utils.data.DataLoader(val_ds, batch_size=32)
# create an instance of our ResNet model
# TODO
resnet = model.ResNet()


# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion = t.nn.BCELoss() # Using BCELoss because model has Sigmoid
optimizer = t.optim.Adam(resnet.parameters(), lr=1e-4, weight_decay=1e-5)

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

trainer = Trainer(
    model=resnet,
    crit=criterion,
    optim=optimizer,
    train_dl=train_dl,
    val_test_dl=val_dl,
    cuda=t.cuda.is_available(),
    early_stopping_patience=10
)


# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()