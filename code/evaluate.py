'''
    Evaluation metrics
    Predicts from model output and prints stats and saves figures.

    2022 Natalie Imirzian
'''

import yaml
import torch
import scipy
import numpy as np
import argparse
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from train import create_dataloader

def load_model(cfg, outdir, epoch=None):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob(outdir+'/model_states/*.pt')

    if len(model_states) > 0:
        # at least one save state found; get latest
        model_epochs = [int(m.replace(outdir+'/model_states/','').replace('.pt','')) for m in model_states]
        if epoch:
            start_epoch = epoch
        else:
            start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Evaluating from epoch {start_epoch}')
        state = torch.load(open(f'{outdir}/model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('No model found')


    return model_instance, start_epoch

def predict(cfg, dataLoader, model):
  with torch.no_grad():   # no gradients needed for prediction
    predictions = []
    predict_labels = []
    labels = []

    for idx, (data, label) in enumerate(dataLoader): 
      prediction = model(data) 
      predict_label = torch.argmax(prediction, dim=1) 

      predictions.append(prediction)
      predict_labels.append(int(predict_label))
      labels.append(int(label))

    return predictions, predict_labels, labels

def save_confusion_matrix(y_true, y_pred, outdir, epoch, split):
    # make figures folder if not there
    os.makedirs(outdir+'/figs', exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(outdir+'/figs/confusion_matrix_epoch'+str(epoch)+'_'+split+'.png', facecolor="white")
    
    return cm

def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--output', required=True, help='Path to output folder')
    parser.add_argument('--split', help='Data split')
    parser.add_argument('--epoch', help='Epoch to load')
    args = parser.parse_args()

    # set model directory
    outdir = args.output

    # get config from model directory
    config = glob(outdir+'*.yaml')[0]

    # load config
    print(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))

    # setup dataloader
    dl_val = create_dataloader(cfg, split=args.split, batch=1)

    # load model and predict from model

    model, epoch = load_model(cfg, outdir, args.epoch)
    predictions, predict_labels, labels = predict(cfg, dl_val, model)   
    
    # get accuracy score
    acc = accuracy_score(labels, predict_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    # confusion matrix
    cm = save_confusion_matrix(labels, predict_labels, outdir, epoch, args.split)

    # precision recall curve

    # save list of predictions



if __name__ == '__main__':
    main()
