import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        output = self._model(x)
        loss = self._crit(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        output = self._model(x)
        loss = self._crit(output, y)
        # Convert probabilities to binary predictions (threshold 0.5)
        preds = (output > 0.5).float()
        return loss.item(), preds

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model.train()
        total_loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            total_loss += self.train_step(x, y)
        return total_loss / len(self._train_dl)

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, preds = self.val_test_step(x, y)
                total_loss += loss
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        avg_loss = total_loss / len(self._val_test_dl)
        # Calculate F1 score (weighted or macro for multi-label)
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        score = f1_score(y_true, y_pred, average='macro')
        print(f"Validation Loss: {avg_loss:.4f}, F1 Score: {score:.4f}")
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO

        while True:

            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            #TODO
            assert self._early_stopping_patience > 0 or epochs > 0

            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            epoch = 0

            while True:
                if epochs > 0 and epoch >= epochs:
                    break

                print(f"Epoch {epoch + 1}")
                train_loss = self.train_epoch()
                val_loss = self.val_test()

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Early Stopping and Checkpointing logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch)
                else:
                    patience_counter += 1

                if 0 < self._early_stopping_patience <= patience_counter:
                    print("Early stopping triggered.")
                    break

                epoch += 1

            return train_losses, val_losses
