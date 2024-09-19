import yaml
import glob
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from dataset import MEARecDataset_full, MEARecDataset_window, MEARecDataset_window_number
from model import backbone, classifier
from utils import train, create_recording, eval


if __name__ == "__main__":
    
    # Read args
    parser = argparse.ArgumentParser(description="fafeSort")
    parser.add_argument('--yaml_file', default='./config/target.yaml', type=str)
    parser.add_argument('--dataset_folder', default='./rec/', type=str)
    parser.add_argument('--run_folder', default='./run/', type=str)
    args = parser.parse_args()

    random.seed(32)
    np.random.seed(32)
    torch.manual_seed(32)
    os.environ["PYTHONHASHSEED"] = str(32)
    torch.cuda.manual_seed_all(32)
    torch.backends.cudnn.deterministic = True

    # Read target yaml file
    with open(args.yaml_file, 'r') as yaml_file:
        target_yaml = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    # Read recordings for pretraining
    target_pretrain_recording_dict = {
        'recording': target_yaml['pretrain']['recording']
    }
    pretrain_recording_file = create_recording(target_rec_dict=target_pretrain_recording_dict,
                                               dataset_folder=args.dataset_folder)
    
    # Pretrain
    pretrain_model_exist = False
    pretrain_model_file = None
    yaml_files = glob.glob(os.path.join(args.run_folder, '**', '*.yaml'), recursive=True)
    for file in yaml_files:
        with open(file, 'r') as yaml_file:
            pretrain_yaml = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        if 'pretrain' in pretrain_yaml and 'finetune' not in pretrain_yaml:
            if (target_yaml['pretrain'] == pretrain_yaml['pretrain']):
                pretrain_model_exist = True
                pretrain_model_file = file.replace('.yaml', '.pt')
                print(f'Pretrained model exists in file \033[92m{pretrain_model_file}\033[0m')
                break
    if not pretrain_model_exist:
        print('\033[91mPretrained model does not exist!\033[0m')
        print(f'\033[92mBegin pretraining model\033[0m')
        if target_yaml['pretrain']['configuration']['full'] == 'True':
            trainset = MEARecDataset_full(recording_file=pretrain_recording_file,
                                          start=0,
                                          end=target_yaml['pretrain']['configuration']['train_duration'],
                                          jitter=5)
        elif target_yaml['pretrain']['configuration']['full'] == 'Number':
            trainset = MEARecDataset_window_number(recording_file=pretrain_recording_file,
                                                   window_number=target_yaml['pretrain']['configuration']['train_duration'],
                                                   jitter=5)
        else:
            trainset = MEARecDataset_window(recording_file=pretrain_recording_file,
                                          start=0,
                                          end=target_yaml['pretrain']['configuration']['train_duration'],
                                          jitter=5)
        testset  = MEARecDataset_full(recording_file=pretrain_recording_file,
                                      start=target_yaml['pretrain']['recording']['duration'] // 2,
                                      end=target_yaml['pretrain']['recording']['duration'],
                                      jitter=5)
        validset = MEARecDataset_full(recording_file=pretrain_recording_file,
                                      start=target_yaml['pretrain']['recording']['duration'] // 2,
                                      end=target_yaml['pretrain']['recording']['duration'],
                                      jitter=0)
        model = nn.Sequential(backbone(),
                              classifier(num_class=trainset.spiketrains.shape[1])).to("cuda")
        
        pretrain_train_acc, pretrain_test_acc = train(model=model,
                                                      train_dataset=trainset,
                                                      test_dataset=testset,
                                                      bs=target_yaml['pretrain']['configuration']['batch'],
                                                      lr=target_yaml['pretrain']['configuration']['learning_rate'],
                                                      epoch=target_yaml['pretrain']['configuration']['epoch'],
                                                      verbose=True)
        eval_result = eval(model, validset)
        
        torch.save(model, f'{args.run_folder}/{len(yaml_files)+1}.pt')
        pretrain_model_file = f'{args.run_folder}/{len(yaml_files)+1}.pt'

        yaml_dict = {
            'pretrain': target_yaml['pretrain'],
            'train_accuracy': pretrain_train_acc.tolist(),
            'test_accuracy': pretrain_test_acc.tolist(),
            'eval_result': eval_result.tolist()
        }

        with open(f'{args.run_folder}/{len(yaml_files)+1}.yaml', 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file)

    if 'finetune' in target_yaml:

        # Read recordings for finetuning
        target_finetune_recording_dict = {
            'recording': target_yaml['finetune']['recording']
        }
        finetune_recording_file = create_recording(target_rec_dict=target_finetune_recording_dict,
                                                   dataset_folder=args.dataset_folder)


        finetune_model_exist = False
        finetune_model_file = None
        yaml_files = glob.glob(os.path.join(args.run_folder, '**', '*.yaml'), recursive=True)
        for file in yaml_files:
            with open(file, 'r') as yaml_file:
                finetune_yaml = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
            if 'finetune' in finetune_yaml:
                if ((target_yaml['finetune'] == finetune_yaml['finetune']) and (target_yaml['pretrain'] == finetune_yaml['pretrain'])):
                    finetune_model_exist = True
                    finetune_model_file = file.replace('.yaml', '.pt')
                    print(f'Finetuned model exists in file \033[92m{finetune_model_file}\033[0m')
                    break
        if not finetune_model_exist:
            print('\033[91mFinetuned model does not exist!\033[0m')
            print(f'\033[92mBegin finetuning model\033[0m')
            if target_yaml['finetune']['configuration']['full'] == 'True':
                trainset = MEARecDataset_full(recording_file=finetune_recording_file,
                                              start=0,
                                              end=target_yaml['finetune']['configuration']['train_duration'],
                                              jitter=5)
            elif target_yaml['finetune']['configuration']['full'] == 'Number':
                trainset = MEARecDataset_window_number(recording_file=finetune_recording_file,
                                                       window_number=target_yaml['finetune']['configuration']['train_duration'],
                                                       jitter=5)
            else:
                trainset = MEARecDataset_window(recording_file=finetune_recording_file,
                                              start=0,
                                              end=target_yaml['finetune']['configuration']['train_duration'],
                                              jitter=5)
            testset  = MEARecDataset_full(recording_file=finetune_recording_file,
                                          start=target_yaml['finetune']['recording']['duration'] // 2,
                                          end=target_yaml['finetune']['recording']['duration'],
                                          jitter=5)
            validset = MEARecDataset_full(recording_file=finetune_recording_file,
                                          start=target_yaml['finetune']['recording']['duration'] // 2,
                                          end=target_yaml['finetune']['recording']['duration'],
                                          jitter=0)
            model = nn.Sequential(torch.load(pretrain_model_file)[0],
                                  classifier(num_class=trainset.spiketrains.shape[1])).to("cuda")
            # model[0].model_fix_para()
            finetune_train_acc, finetune_test_acc = train(model=model,
                                                          train_dataset=trainset,
                                                          test_dataset=testset,
                                                          bs=target_yaml['finetune']['configuration']['batch'],
                                                          lr=target_yaml['finetune']['configuration']['learning_rate'],
                                                          epoch=target_yaml['finetune']['configuration']['epoch'],
                                                          verbose=True)
            
            eval_result = eval(model, validset)
            
            torch.save(model, f'{args.run_folder}/{len(yaml_files)+1}.pt')
            finetune_model_file = f'{args.run_folder}/{len(yaml_files)+1}.pt'

            yaml_dict = {
                'pretrain': target_yaml['pretrain'],
                'finetune': target_yaml['finetune'],
                'train_accuracy': finetune_train_acc.tolist(),
                'test_accuracy': finetune_test_acc.tolist(),
                'eval_result': eval_result.tolist()
            }
    
            with open(f'{args.run_folder}/{len(yaml_files)+1}.yaml', 'w') as yaml_file:
                yaml.dump(yaml_dict, yaml_file)
