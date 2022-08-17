'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger, edited by Natalie Imirzian
'''

import os
import argparse
import yaml
import glob
from tqdm import trange
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim import lr_scheduler # add learning rate scheduling
from torch.utils.tensorboard import SummaryWriter 

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# let's import our own classes and functions!
from dataset import SizeDataset, Transform
from model import CustomResNet18


def create_dataloader(cfg, split='train', transforms=None, batch=None):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    if batch==None:
        batch = cfg['batch_size']

    #transforms = Transform()
    dataset_instance = SizeDataset(cfg, split, transform=transforms)        
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=batch,
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader



def create_outdir(cfg, folder):
    '''
        Creates a folder for experiment with config file
    '''
    # create folder
    os.makedirs(folder, exist_ok=True)
    
    # copy config file to save model settings
    shutil.copy(cfg, folder)




def save_confusion_matrix(y_true, y_pred, acc, outdir, epoch, split):
    # make figures folder if not there
    os.makedirs(outdir+'/figs', exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Accuracy: {:.2f}".format(acc))
    plt.savefig('{}/figs/confusion_matrix_epoch{:02d}_{}.png'.format(outdir, epoch, split), facecolor="white")
    plt.clf() # clear plot to reduce memory usages
    
    return cm



def load_model(cfg, outdir):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])  # create an object instance of our CustomResNet18 class
           
    # load latest model state
    model_states = glob.glob(outdir+'/model_states/*.pt')
    if len(model_states) > 0:
        # at least one save state found; get latest
        model_epochs = [int(m.replace(outdir+'/model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'{outdir}/model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(epoch, model, stats, outdir):
    # make sure save directory exists; create if not
    os.makedirs(outdir+'/model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'{outdir}/model_states/{epoch}.pt', 'wb'))



def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def get_fuzzy_accuracy(y_true, y_pred):
    facc = 0
    for true, pred in zip(y_true, y_pred):
        if pred in range(true - 1, true + 1, 1):
            facc += 1
    
    facc /= len(y_true)

    return facc



def train(cfg, dataLoader, model, optimizer, epoch, outdir):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # running averages
    loss_total, oa_total, fa_total = 0.0, 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))

    # create list for labels
    true_labels = []
    pred_labels = []

    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label
        #step = idx + (epoch - 1)*idx

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        true_labels.extend(labels.numpy().tolist())
        pred_labels.extend(pred_label.numpy().tolist())

        # fuzzy accuracy
        
        #fa = torch.mean((pred_label == labels-1).float())
        #fa_total += fa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    #import IPython
    #IPython.embed()
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           
    writer.add_scalar("Loss/train", loss_total, epoch)
    oa_total /= len(dataLoader)
    writer.add_scalar("Acc/train", oa_total, epoch)
    fa = get_fuzzy_accuracy(true_labels, pred_labels)
    writer.add_scalar("Fa/val", fa, epoch)

    # save confusion matrix
    save_confusion_matrix(true_labels, pred_labels, oa_total, outdir, epoch, "train")

    return loss_total, oa_total



def validate(cfg, dataLoader, model, epoch, outdir):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))

    # create label list
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            true_labels.extend(labels.numpy().tolist())
            pred_labels.extend(pred_label.numpy().tolist())

            progressBar.set_description(
                '[Val] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    writer.add_scalar("Loss/val", loss_total, epoch)
    oa_total /= len(dataLoader)
    writer.add_scalar("Acc/val", oa_total, epoch)
    fa = get_fuzzy_accuracy(true_labels, pred_labels)
    writer.add_scalar("Fa/val", fa, epoch)

    # save confusion matrix
    save_confusion_matrix(true_labels, pred_labels, oa_total, outdir, epoch, "val")

    return loss_total, oa_total



def main():
    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='../configs/ant_size.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # show model progress on tensorboard
    writer = SummaryWriter(comment=cfg['experiment'])

   #print(f'Saving results to {cfg['experiment']}')
    outdir = os.path.join('/datadrive/experiments/', cfg['experiment'])
    create_outdir(args.config, outdir)

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # create tranformation
    transforms = A.Compose([
        A.Rotate(-cfg['rotate_deg'], cfg['rotate_deg']),
        A.Flip(p=cfg['flip_prob']),
        A.ColorJitter(p=cfg['color_jitter']),
        A.CoarseDropout(p=cfg['dropout'], max_height=15, max_width=15, max_holes=4),
        A.GaussNoise(p=cfg['noise']),
        A.ToSepia(p=cfg['sepia']),
        A.ToFloat(max_value=255.0),  
        ToTensorV2()
    ])

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train', transforms=transforms)
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg, outdir)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim, current_epoch, outdir)
        loss_val, oa_val = validate(cfg, dl_val, model, current_epoch, outdir)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(current_epoch, model, stats, outdir)



if __name__ == '__main__':
    main()
