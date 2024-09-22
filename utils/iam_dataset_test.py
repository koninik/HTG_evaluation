import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize, centered
import os

class IAMDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms, args):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, args)
        self.setname = 'IAM'
        self.args = args
        self.trainset_file = '{}/{}/set_split/trainset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/set_split/validationset1.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/set_split/testset.txt'.format(self.basefolder, self.setname)
        self.line_file = '/home/x_konni/Desktop/x_konni/datasets/IAM/ascii/lines.txt'.format(self.basefolder, self.setname)
        self.word_file = '/home/x_konni/Desktop/x_konni/datasets/IAM/ascii/words.txt' #.format(self.basefolder, self.setname)
        self.word_path = self.basefolder #.format(self.basefolder, self.setname)
        self.line_path = self.basefolder #'{}/lines'.format(self.basefolder, self.setname)
            
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(self, set='train', level='word'):
            if subset == 'train':
                #valid_set = np.loadtxt(self.trainset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/train.uttlist', dtype=str)
                #print(valid_set)
            elif subset == 'val':
                #valid_set = np.loadtxt(self.valset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/validation.uttlist', dtype=str)
            elif subset == 'test':
                #valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/test.uttlist', dtype=str)
            else:
                raise ValueError
            if level == 'word':
                gtfile= self.word_file
                root_path = self.word_path
                print('root_path', root_path)
            elif level == 'line':
                gtfile = self.line_file
                root_path = self.line_path
            else:
                raise ValueError
            gt = []
            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]
                    name_parts = name.split('-')
                    pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
                    if level == 'word':
                        line_name = pathlist[-2]
                        del pathlist[-2]
                    elif level == 'line':
                        line_name = pathlist[-1]
                    form_name = '-'.join(line_name.split('-')[:-1])
                    
                    #if (info[1] != 'ok') or (form_name not in valid_set):
                    
                    if (level == 'word') and (info[1] != 'ok'):
                        continue
                    
                    if (form_name not in valid_set):
                        #print(line_name)
                        continue
                    img_path = '/'.join(pathlist)
                    transcr = ' '.join(info[8:])
                    gt.append((img_path, transcr))
            return gt

        
        def gather_augmented_info(self, aug_data_path, gtfile):
            gt_aug = []
            with open(gtfile, 'r') as f:
                train_data = f.readlines()
                train_data = [i.strip().split(',') for i in train_data]
                print('augmented data', len(train_data))
                for i in train_data:
                    
                    #s_id = i[0].split(',')[0]
                    
                    image = i[0] #+ '.png'

                    if image.endswith('.png'):
                        image = image
                    else:
                        image = image + '.png'

                    transcr = i[2]
                    
                    img_path = os.path.join(aug_data_path, image)
                    #print('img_path', img_path)
                    gt_aug.append((img_path, transcr))
            return gt_aug
        data = []
        if subset == 'train':
            info = gather_iam_info(self, subset, segmentation_level)
            info = info 
        else:
            synth_data_dir = self.args.synth_data_img
            print('synth_data_dir', synth_data_dir)
            synth_gt_file = self.args.synth_data_txt
            print('synth_gt_file', synth_gt_file)

            info = gather_augmented_info(self, synth_data_dir, synth_gt_file)
        
        for i, (img_path, transcr) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            #

            try:
                if img_path.endswith('.png'):
                    
                    img = img_io.imread(img_path)
                else:
                    img = img_io.imread(img_path + '.png') #original code
                
                img = 1 - img.astype(np.float32) / 255.0
                
            except:
                continue
                
            #except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")

            data += [(img, transcr)]
        
        
        
        return data

        
        