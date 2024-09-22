import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
from utils.auxilary_functions import *
import timm
import time
import json


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

class WordStyleDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)   
    '''
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    '''
    def __init__(self, 
        basefolder: str = 'datasets/',                #Root folder
        subset: str = 'all',                          #Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = 'line',             #Type of data to load ('line' or 'word')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        data_file = './htg_style_test_split.3.txt'
        ):
        
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None                             # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.max_transcr_len = 0
        self.data_file = data_file

        with open(self.data_file, 'r') as f:
            lines = f.readlines()
        wid_dict = './writers_dict.json'
        with open(wid_dict, 'r') as f:
            self.wid_dict = json.load(f)
        
        self.data_info = [line.strip().split(',') for line in lines]
        
    def __len__(self):
        return len(self.data_info)

   
    def __getitem__(self, index):
        
        #if img ends with .png leave it as it is, otherwise add .png
        img = self.data_info[index][0]
        if img.endswith('.png'):
            img = img
        else:
            img = img + '.png'
        
        img_path = os.path.join(self.basefolder, img) 
        img = Image.open(img_path).convert('RGB')
        transcr = self.data_info[index][2]

        wid = self.data_info[index][1]
        wid = self.wid_dict[wid]
        wid = torch.tensor(int(wid)).to(torch.int64)
        
        if self.transforms is not None:
            
            img = self.transforms(img)
            
        return img, transcr, wid 

    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        img, transcr, wid = zip(*batch)

        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        wid = torch.stack(wid)
        return images_batch, transcr, wid 


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )
        #self.model = torch.compile(self.model, backend="inductor")
        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        x = self.model(x)
        return x       
    

#================ Performance and Loss Function ========================
def performance(pred, label):
    loss = nn.CrossEntropyLoss()
    loss = loss(pred, label)
    return loss 

#===================== Training ==========================================

def train_class_epoch(model, training_data, optimizer, args):
    '''Epoch operation in training phase'''
    
    model.train()
    total_loss = 0
    n_corrects = 0 
    total = 0
    pbar = tqdm(training_data)
    for i, data in enumerate(pbar):
    
        image = data[0]
        if args.dataset == 'iam':
            label = data[2]  
        else:
            label = data[1]
        
        image = image.to(args.device)
        label = label.to(args.device)
        
        optimizer.zero_grad()
        
        output = model(image)
        
        loss = performance(output, label)
        _, preds = torch.max(output.data, 1)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        total += label.size(0)
        n_corrects += (preds == label).sum().item()
        pbar.set_postfix(Loss=loss.item())
        
    loss = total_loss/total
    accuracy = n_corrects/total
    
    return loss, accuracy

def eval_class_epoch(model, validation_data, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0
    prediction_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_data)):

            image = data[0]   

            if args.dataset == 'iam':
                label = data[2]  
            else:
                label = data[1]
            
            label = data[2] 
            image = image.to(args.device)
            label = label.to(args.device)

            output = model(image)
            
            loss = performance(output, label)  #performance
            _, preds = torch.max(output.data, 1)
            
            total_loss += loss.item()
            n_corrects += (preds == label.data).sum().item()
            total += label.size(0)
            #prediction_list.append(preds)
            #write into a file the img_path and the prediction
            # with open('predictions.txt', 'a') as f:
            #     for i, p in enumerate(preds):
            #         f.write(f'{image_paths[i]},{p}\n')
            
    loss = total_loss/total
    accuracy = n_corrects/total

    return loss, accuracy


def train_classification(model, training_data, validation_data, optimizer, scheduler, device, args): #scheduler # after optimizer
    ''' Start training '''

    num_of_no_improvement = 0
    best_acc = 0
    
    for epoch_i in range(args.epochs):
        print('[Epoch', epoch_i, ']')

        start = time.time()
        
        train_loss, train_acc = train_class_epoch(model, training_data, optimizer, args)
        print('Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100*train_acc,
                  elapse=(time.time()-start)/60))
        
        start = time.time()
        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}

        if validation_data is not None:
            val_loss, val_acc = eval_class_epoch(model, validation_data, args)
            print('Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                        loss=val_loss, accu=100*val_acc,
                    elapse=(time.time()-start)/60))
            
            if val_acc > best_acc:
                
                print('- [Info] The checkpoint file has been updated.')
                best_acc = val_acc
                
                torch.save(model.state_dict(), "./HTG_style_model_new.pth")
                
                num_of_no_improvement = 0
            else:
                num_of_no_improvement +=1
            
        
            if num_of_no_improvement >= 10:
                        
                print("Early stopping criteria met, stopping...")
                break
        else:
            torch.save(model.state_dict(), "./HTG_style_model_new.pth")
        
        scheduler.step()
        
    


def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Document Classification')
    parser.add_argument('--model', type=str, default='resnet18', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--dataset', type=str, default='iam', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--batch_size', type=int, default=224, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, required=False, help='number of training epochs')
    parser.add_argument('--pretrained', type=bool, default=False, help='keep False to test the generated data')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--style_model_path', type=str, default='./trained_models/HTG_style_model.pth', help='path to style models')
    parser.add_argument('--synth_data_path', type=str, default='/path/to/synthetic/data', help='path to save models')
    parser.add_argument('--mode', type=str, default='classification', help='triplet or classification')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print('Using resnet18')
    model = ImageEncoder(model_name=args.model, num_classes=339, pretrained=True, trainable=True)
    print('Model loaded')
    
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.pretrained == True:
        PATH = args.style_model_path
        
        state_dict = torch.load(PATH, map_location=args.device)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        
    model = model.to(device)
    #print(model)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    
    if args.mode == 'classification':
        train = False
        new_transf = transforms.Compose([
                            transforms.Resize(64),
                            transforms.CenterCrop((64, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        if train == True:

            iam_folder = 'path/to/IAM/words'
            
            train_data = WordStyleDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), transforms=new_transf, data_file='./htg_style_train_split.txt')
            val_data = WordStyleDataset(iam_folder, 'val', 'word', fixed_size=(1 * 64, 256), transforms=new_transf, data_file='./htg_style_val_split.txt')

            print('Length of train data', len(train_data))
            print('Length of val data', len(val_data))
            
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

            train_classification(model, train_loader, val_loader, optimizer_ft, scheduler, device, args)
            print('finished training')
        else:
            
            print(f'Testing Writer Identification with {args.model}')
            test_dataset_folder = args.synth_data_path
            
            test_data = WordStyleDataset(test_dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), transforms=new_transf, data_file='./htg_style_test_split.txt')
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
            print('Test data', len(test_data))
            val_loss, val_acc = eval_class_epoch(model, test_loader, args)
            print('Test: {loss: 8.5f} , accuracy: {accu:3.3f}'.format(loss=val_loss, accu=100*val_acc))
    
    
if __name__ == '__main__':
    main()