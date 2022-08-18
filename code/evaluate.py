'''
    Evaluation metrics
    Predicts from model output and prints stats and saves figures.

    2022 Natalie Imirzian
'''

import yaml
import torch
import torch.nn as nn
import pandas as pd

import argparse
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from model import CustomResNet18

from torch.utils.data import DataLoader
from dataset import SizeDataset

def load_model(cfg, outdir, epoch=None):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob(outdir+'/model_states/*.pt')

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

        #import IPython
        #IPython.embed()

    else:
        # no save state found; start anew
        print('No model found')


    return model_instance, start_epoch

def predict(dataLoader, model):
  with torch.no_grad():   # no gradients needed for prediction
    predictions = []
    predict_labels = []
    labels = []
    filename = []

    model.eval()

    for idx, (data, label) in enumerate(dataLoader): 
      prediction = model(data) 
      predict_label = torch.argmax(prediction, dim=1) 

      predictions.append(prediction)
      predict_labels.append(int(predict_label))
      labels.append(int(label))
      filename.append(dataLoader.dataset.data[idx][0])

    return filename, predictions, predict_labels, labels



def get_fuzzy_accuracy(y_true, y_pred):
    facc = 0
    for true, pred in zip(y_true, y_pred):
        if pred in range(true - 1, true + 1, 1):
            facc += 1
    
    facc /= len(y_true)

    return facc



def save_un_confusion_matrix(y_true, y_pred, outdir, epoch, split):
    # make figures folder if not there
    os.makedirs(outdir+'/figs', exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(outdir+'/figs/confusion_matrix_epoch'+str(epoch)+'_'+str(split)+'.png', facecolor="white")
    
    return cm



def create_results(files, predictions, y_true, y_pred):
    sm = []
    
    for p in predictions:
        sm.append(nn.Softmax(p))

    df = pd.DataFrame({'filename': files,
                    'likelihood': sm,
                    'predict_label': y_pred,
                    'real_label': y_true}
    )
    
    return df



def get_model_outputs(outdir):
    keys = ['epoch', 'loss_train', 'loss_val', 'oa_train', 'oa_val']
    output_dict = {k: [] for k in keys}

    model_states = glob(outdir+'/model_states/*.pt')

    for m in model_states:
        state = torch.load(open(m, 'rb'))
        output_dict['epoch'].append(int(os.path.basename(m).split('.')[0]))
        output_dict['loss_train'].append(state['loss_train'])
        output_dict['loss_val'].append(state['loss_val'])
        output_dict['oa_train'].append(state['oa_train'])
        output_dict['oa_val'].append(state['oa_val'])

    output_dict['epoch'].sort()
    
    return output_dict



def save_confusion_matrix(y_true, y_pred, acc, outdir, epoch, split):
    # make figures folder if not there
    os.makedirs(outdir+'/figs', exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Accuracy: {:.2f}".format(acc))
    plt.savefig('{}/figs/confusion_matrix_epoch{}_{}.png'.format(outdir, epoch, split), facecolor="white")
    plt.clf() # clear plot to reduce memory usages
    
    return cm



def save_acc_plot(results, outdir):
    os.makedirs(outdir+'/figs', exist_ok=True)

    plt.rcParams['figure.figsize'] = [8, 5]
    plt.rcParams['figure.dpi'] = 300    
    plt.plot(results['epoch'], results['oa_train'], color="blue", label="train")
    plt.plot(results['epoch'], results['oa_val'], color="red", label="val")
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('overall accuracy')
    plt.legend()
    
    plt.savefig(outdir+'/figs/accuracy.png', facecolor="white")

    plt.close()


def save_loss_plot(results, outdir):
    os.makedirs(outdir+'/figs', exist_ok=True)

    plt.rcParams['figure.figsize'] = [8, 5]
    plt.rcParams['figure.dpi'] = 300    

    plt.plot(results['epoch'], results['loss_train'], color="blue", label="train")
    plt.plot(results['epoch'], results['loss_val'], color="red", label="val")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.savefig(outdir+'/figs/loss.png', facecolor="white")

    plt.close()


def main():
    # Argument parser for command-line arguments:
    # python code/train.py --output model_runs
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--output', required=True, help='Path to output folder')
    parser.add_argument('--split', help='Data split', default='val')
    parser.add_argument('--epoch', help='Epoch to load')
    args = parser.parse_args()

    # set model directory
    outdir = args.output

    # get config from model directory
    config = glob(outdir+'/*.yaml')[0]

    # load config
    print(f'Using config "{config}" and "{args.split}" set')
    cfg = yaml.safe_load(open(config, 'r'))

    # get model outputs across epochs
    #out_dic = get_model_outputs(outdir)
    #save_loss_plot(out_dic, outdir)
    #save_acc_plot(out_dic, outdir)


    # setup dataloader
    dataset_instance = SizeDataset(cfg, split=args.split)
    dl = DataLoader(
        dataset=dataset_instance,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    
    

    # load model and predict from model
    model, epoch = load_model(cfg, outdir, args.epoch)
    fn, predictions, predict_labels, labels = predict(dl, model)   
    
    # get accuracy score
    acc = accuracy_score(labels, predict_labels)
    print("Accuracy of model is {:0.2f}".format(acc))
    
    # get fuzzy accuracy
    facc = get_fuzzy_accuracy(labels, predict_labels)
    print("Accuracy within 1 class is {:0.2f}".format(facc))

    # confusion matrix
    cm = save_confusion_matrix(labels, predict_labels, acc, outdir, epoch, args.split)

    # save list of predictions with filename
    df = create_results(fn, predictions, labels, predict_labels)
    df.to_csv(outdir+'/results_epoch'+str(epoch)+'_'+str(args.split)+'.csv', index = False)

    # precision recall curve



if __name__ == '__main__':
    main()
