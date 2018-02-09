import random
import argparse
import glob
import re
import csv
from os.path import join
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm
from PIL import Image
from conditional import conditional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, sampler

from utils import *
from train_utils import *
from custom_dataset import IEEECameraDataset, preprocess_image
from custom_scheduler import ReduceLROnPlateau,EarlyStopping
from custom_models import DyiNet
import logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=400, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=24, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-p', '--pool', type=str, default='max', help='Batch Size during training, e.g. -b 64')
parser.add_argument('-clw', '--center_loss-weight', type=float, default=0.5, help='loss weight')
parser.add_argument('-cla', '--center_loss-alpha', type=float, default=0.01, help='loss alpha')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', type=str, default=None,help='load hdf5 model including weights (and continue training)')
parser.add_argument('-e', '--embedding_size', type=int, default=128, help='featrue dim')
parser.add_argument('-s', '--start_epoch', type=int, default=1, help='featrue dim')

#parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
#parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for FC layers')
#parser.add_argument('-doc', '--dropout-classifier', type=float, default=0., help='Dropout rate for classifier')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-cs', '--crop-size', type=int, default=224, help='Crop size')
parser.add_argument('-w', '--workers', type=int, default=12, help='Num workers')
#parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
#parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
#parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
#parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
#parser.add_argument('-lkf', '--learn-kernel-filter', action='store_true', help='Add a trainable kernel filter before classifier')
#parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
#parser.add_argument('-cl', '--use-center-loss', action='store_true', help='Use center loss')
parser.add_argument('-x', '--extra-dataset', action='store_true', help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')
#parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')
parser.add_argument('-es', '--ensembling', type=str, default='arithmetic', help='Type of ensembling: arithmetic|geometric for TTA')
parser.add_argument('-tta', action='store_true', help='Enable test time augmentation')

args = parser.parse_args()

num_workers = args.workers

#TRAIN_FOLDER       = '../sp-society-camera-model-identification-master/train'
#EXTRA_TRAIN_FOLDER = '../sp-society-camera-model-identification-master/flickr_images' #not used -> ./good_imgs_train.txt
#EXTRA_VAL_FOLDER   = '../sp-society-camera-model-identification-master/val_images' #not used -> ./good_imgs_val.txt
#TEST_FOLDER        = '../sp-society-camera-model-identification-master/test'

TRAIN_FOLDER       = 'train'
EXTRA_TRAIN_FOLDER = 'flickr_images' #not used -> ./good_imgs_train.txt
EXTRA_VAL_FOLDER   = 'val_images' #not used -> ./good_imgs_val.txt
TEST_FOLDER        = 'test'

CROP_SIZE = args.crop_size

m_names = 'nasnetalarge'

# MAIN
if args.model is not None:
    print("Loading model " + args.model)

    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']+1
    model = DyiNet(args.embedding_size, args.pool,len(CLASSES),model_name=m_names,  pretrained=args.use_imagenet_weights, checkpoint=checkpoint)
    
    # e.g. DenseNet201_do0.3_doc0.0_avg-epoch128-val_acc0.964744.hdf5
    #args.classifier = match.group(2)
    #CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[0][1]
else:
    model = DyiNet(args.embedding_size,args.pool, len(CLASSES), model_name=m_names, pretrained=args.use_imagenet_weights)
if cuda_is_available:
    #model = nn.DataParallel(model).cuda()
    model = model.cuda()

if not (args.test or args.test_train):

    # TRAINING
    ids = glob.glob(join(TRAIN_FOLDER, '*/*.jpg'))
    ids.sort()

    if not args.extra_dataset:
        ids_train, ids_val = train_test_split(ids, test_size=0.1, random_state=SEED)
    else:
        #ids_train = [line.rstrip('\n') for line in open('good_imgs_train.txt')]
        #ids_val   = [line.rstrip('\n') for line in open('good_imgs_val.txt')]
        ids_train = ids
        ids_val   = [ ]
        
        extra_train_ids = [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n')) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'good_jpgs'))]
        low_quality =     [os.path.join(EXTRA_TRAIN_FOLDER,line.rstrip('\n').split(' ')[0]) \
            for line in open(os.path.join(EXTRA_TRAIN_FOLDER, 'low-quality'))]
        extra_train_ids = [idx for idx in extra_train_ids if idx not in low_quality]
        extra_train_ids.sort()
        ids_train.extend(extra_train_ids)
        random.shuffle(ids_train)
       
        extra_val_ids = glob.glob(join(EXTRA_VAL_FOLDER,'*/*.jpg'))
        extra_val_ids.sort()
        ids_val.extend(extra_val_ids)

        classes_val = [get_class(idx.split('/')[-2]) for idx in ids_val]
        classes_val_count = np.bincount(classes_val)
        max_classes_val_count = max(classes_val_count)

        # Balance validation dataset by filling up classes with less items from training set (and removing those from there)
        for class_idx in range(N_CLASSES):
            idx_to_transfer = [idx for idx in ids_train \
                if get_class(idx.split('/')[-2]) == class_idx][:max_classes_val_count-classes_val_count[class_idx]]

            ids_train = list(set(ids_train).difference(set(idx_to_transfer)))

            ids_val.extend(idx_to_transfer)

        #random.shuffle(ids_val)
    
    print("Training set distribution:")
    print_distribution(ids_train)

    print("Validation set distribution:")
    print_distribution(ids_val)
    
    classes_train = [get_class(idx.split('/')[-2]) for idx in ids_train]
    class_weight = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)
    classes_val = [get_class(idx.split('/')[-2]) for idx in ids_val]
    
    weights = [class_weight[i_class] for i_class in classes_train]
    weights = torch.DoubleTensor(weights)
    train_sampler = sampler.WeightedRandomSampler(weights, len(weights))
    
    weights = [class_weight[i_class] for i_class in classes_val]
    weights = torch.DoubleTensor(weights)
    val_sampler = sampler.WeightedRandomSampler(weights, len(weights))
    
    train_dataset = IEEECameraDataset(ids_train, crop_size=CROP_SIZE, training=True, model=m_names)
    val_dataset = IEEECameraDataset(ids_val, crop_size=CROP_SIZE, training=False, model=m_names)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size // 4, sampler=val_sampler, num_workers=num_workers, pin_memory=True, collate_fn=default_collate_unsqueeze)
    
    criterion = nn.CrossEntropyLoss()
    #criterion1 = CenterLoss(10,2,args.loss_weight)
    #criterion = [criterion0,criterion1]  

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer1 = optim.Adam(criterion1.parameters(), lr=args.learning_rate)
    #optimizer = [optimizer0,optimizer1]

    scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=5, min_lr=1e-9, epsilon=1e-5, verbose=1, mode='min')
    looker = EarlyStopping() 
    best_val_loss = None
    train_and_validate(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        looker,
        criterion,
        args.max_epoch,
        args.start_epoch,
        best_val_loss,
        args.center_loss_alpha,
        args.center_loss_weight,
        m_names,

    )
else:
    # TEST
    if args.test:
        ids = glob.glob(join(TEST_FOLDER, '*.tif'))
    elif args.test_train:
        ids = glob.glob(join(TRAIN_FOLDER, '*/*.jpg'))
    else:
        assert False

    ids.sort()

    match = re.search(r'([^/]*)\.pth', args.model)
    model_name = match.group(1) + ('_tta_' + args.ensembling if args.tta else '')
    csv_name   = 'submission_' + model_name + '.csv'
    
    model.eval()
    with conditional(args.test, open(csv_name, 'w')) as csvfile:

        if args.test:
            csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['fname', 'camera'])
            classes = []
            predicts = []
        else:
            correct_predictions = 0

        for i, idx in enumerate(tqdm(ids)):

            img = np.array(Image.open(idx))

            if args.test_train:
                img = get_crop(img, 512*2, random_crop=False)

            original_img = img

            original_manipulated = np.float32([1. if idx.find('manip') != -1 else 0.])

            sx = img.shape[1] // CROP_SIZE
            sy = img.shape[0] // CROP_SIZE

            if args.test and args.tta:
                transforms = [[], ['orientation']]
            elif args.test_train:
                transforms = [[], ['orientation'], ['manipulation'], ['manipulation', 'orientation']]
            else:
                transforms = [[]]

            img_batch         = np.zeros((len(transforms)* 4 * 4, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
            manipulated_batch = np.zeros((len(transforms)* 4 * 4, 1),  dtype=np.float32)
            
            i = 0
            for transform in transforms:
                img = np.copy(original_img)
                manipulated = np.copy(original_manipulated)

                if 'orientation' in transform:
                    img = np.rot90(img, 1, (0,1))
                if 'manipulation' in transform and not original_manipulated:
                    img = random_manipulation(img)
                    manipulated = np.float32([1.])

                if args.test_train:
                    img = get_crop(img, 512, random_crop=False)

                sx = img.shape[1] // CROP_SIZE
                sy = img.shape[0] // CROP_SIZE

                #for x in range(sx):
                #    for y in range(sy):
                for x in [0,.4,.8,1.2]:
                    for y in [0,.4,.8,1.2]:
                        _img = np.copy(img[int(y*CROP_SIZE):int((y+1)*CROP_SIZE), int(x*CROP_SIZE):int((x+1)*CROP_SIZE)])
                        img_batch[i]         = preprocess_image(_img, m_names)
                        manipulated_batch[i] = manipulated
                        i += 1
            predictions = np.zeros((32,10),dtype=np.float32)
            for j in range(4): 
                img_batchj, manipulated_batchj = variable(torch.from_numpy(img_batch[j*8:(j+1)*8])), variable(torch.from_numpy(manipulated_batch[j*8:(j+1)*8]))
                prediction = model(img_batchj, manipulated_batchj).data.cpu().numpy()
                if prediction.shape[0] != 1: # TTA
                    if args.ensembling == 'geometric':
                        prediction = np.log(prediction + K.epsilon()) # avoid numerical instability log(0)
                predictions[j*8:(j+1)*8] = prediction	
            prediction = np.sum(predictions, axis=0)
            predicts.append(prediction) 
            prediction_class_idx = np.argmax(prediction)

            if args.test_train:
                class_idx = get_class(idx.split('/')[-2])
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:
                csv_writer.writerow([idx.split('/')[-1], CLASSES[prediction_class_idx]])
                classes.append(prediction_class_idx)
        
        if args.test_train:
            print("Accuracy: " + str(correct_predictions / (len(transforms) * i)))
        
        if args.test:
            np.save('{}.npy'.format(model_name), predicts)
            print("Test set predictions distribution:")
            print_distribution(None, classes=classes)
            print("Now you are ready to:")
            print("kg submit {}".format(csv_name))
