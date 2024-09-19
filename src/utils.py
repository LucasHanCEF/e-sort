import os
import glob
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import MEAutility as MEA
import MEArec as mr
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
from dataset import MEARecDataset_full


def create_recording(target_rec_dict: dict,
                     dataset_folder: str,
                     config_path='./config'):
    target_template_dict = {
        'template': {
            'probe': target_rec_dict['recording']['probe'],
            'N' : target_rec_dict['recording']['N'],
            'drifting': target_rec_dict['recording']['drifting']
        }
    }
    MEA.add_mea(f'{config_path}/Neuropixels-Ultra.yaml')
    template_file = create_template(target_template_dict=target_template_dict,
                                    dataset_folder=dataset_folder)
    recording_exist = False
    recording_file = None
    yaml_files = glob.glob(os.path.join(dataset_folder, '**', '*.yaml'), recursive=True)
    for file in yaml_files:
        with open(file, 'r') as yaml_file:
            recording_yaml = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        if 'recording' in recording_yaml:
            if target_rec_dict['recording'] == recording_yaml['recording'] :
                recording_exist = True
                recording_file = file.replace('.yaml', '.h5')
                print(f'Recording exists in file \033[92m{recording_file}\033[0m')
                break
    if not recording_exist:
        print('\033[91mRecording does not exist!\033[0m')
        print('\033[92mBegin generating recording!\033[0m')
        recording_params = mr.get_default_recordings_params()

        tempgen = mr.load_templates(template_file)
        tempgen_info = tempgen.info['params']

        sqmea = MEA.return_mea(target_rec_dict['recording']['probe'])
        x_range = tempgen_info['xlim'][1] - tempgen_info['xlim'][0]
        y_range = sqmea.pitch[0] * sqmea.dim[0] + sqmea.size + tempgen_info['overhang']
        z_range = sqmea.pitch[1] * sqmea.dim[1] + sqmea.size + tempgen_info['overhang']
        volume = x_range * y_range * z_range
        n = target_rec_dict['recording']['density'] * volume * 1e-9

        # Set parameters
        recording_params['spiketrains']['n_exc'] = int(n*0.8)
        recording_params['spiketrains']['n_inh'] = int(n*0.2)
        recording_params['spiketrains']['duration'] = target_rec_dict['recording']['duration']

        print(f'Generate {recording_params['spiketrains']['duration']} seconds recording with {recording_params['spiketrains']['n_exc']}/{recording_params['spiketrains']['n_inh']} number of excitatory/inhibitory neurons')

        recording_params['templates']['min_amp'] = 40
        recording_params['templates']['max_amp'] = 300
        recording_params['templates']['min_disc'] = 5

        recording_params['recordings']['noise_level'] = target_rec_dict['recording']['noise']

        # use chunk options
        recording_params['recordings']['chunk_conv_duration'] = 10
        recording_params['recordings']['chunk_noise_duration'] = 10
        recording_params['recordings']['chunk_filter_duration'] = 10

        # drifting option
        if target_rec_dict['recording']['drifting'] != 'None':
            recording_params["recordings"]["drifting"] = True
            if target_rec_dict['recording']['drifting'] == 'slow':
                recording_params["recordings"]["drift_mode_probe"] = 'rigid'
                recording_params["recordings"]["drift_fs"] = 5
                recording_params["recordings"]["slow_drift_velocity"] = 10
                recording_params["recordings"]["slow_drift_amplitude"] = 30
                print(f'Rigid slow drifting is applied')
            elif target_rec_dict['recording']['drifting'] == 'fast':
                recording_params["recordings"]["drift_mode_speed"] = "fast"
                recording_params["recordings"]["fast_drift_period"] = 60
                recording_params["recordings"]["fast_drift_max_jump"] = 15
                print(f'Rigid fast drifting is applied')
            elif target_rec_dict['recording']['drifting'] == 'nonrigid':
                recording_params["recordings"]["drift_mode_probe"] = "non-rigid"
                recording_params["recordings"]["drift_mode_speed"] = "slow"
                recording_params["recordings"]["slow_drift_waveform"] = "sine"
                recording_params["recordings"]["slow_drift_velocity"] = 80
                recording_params["recordings"]["slow_drift_amplitude"] = 10
                print(f'Non-rigid fast sine oscillation drifting is applied')
        else:
            print(f'Dirfting is not applied')


        recording_params['spiketrains']['seed'] = target_rec_dict['recording']['seed']
        recording_params['templates']['seed'] = target_rec_dict['recording']['seed']
        recording_params['recordings']['seed'] = target_rec_dict['recording']['seed']
        recording_params['seeds']['spiketrains'] = target_rec_dict['recording']['seed']
        recording_params['seeds']['templates'] = target_rec_dict['recording']['seed']
        recording_params['seeds']['convolution'] = target_rec_dict['recording']['seed']
        recording_params['seeds']['noise'] = target_rec_dict['recording']['seed']

        recgen = mr.gen_recordings(tempgen=tempgen, params=recording_params, verbose=True, n_jobs=32)
        mr.save_recording_generator(recgen, filename=f'{dataset_folder}/{len(yaml_files)+1}.h5')
        with open(f'{dataset_folder}/{len(yaml_files)+1}.yaml', 'w') as yaml_file:
            yaml.dump(target_rec_dict, yaml_file)
        recording_file = f'{dataset_folder}/{len(yaml_files)+1}.h5'
    return recording_file



def create_template(target_template_dict: dict,
                    dataset_folder: str):
    template_exist = False
    template_file = None
    yaml_files = glob.glob(os.path.join(dataset_folder, '**', '*.yaml'), recursive=True)
    for file in yaml_files:
        with open(file, 'r') as yaml_file:
            template_yaml = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        if 'template' in template_yaml:
            if target_template_dict['template'] == template_yaml['template'] :
                template_exist = True
                template_file = file.replace('.yaml', '.h5')
                print(f'Template for recording exists in file \033[92m{template_file}\033[0m')
                break
    if not template_exist:
        print('\033[91mTemplate for recording does not exist!\033[0m')
        # Get default params
        templates_params = mr.get_default_templates_params()
        cell_models_folder = mr.get_default_cell_models_folder()

        # Define probe and 
        templates_params['probe'] = target_template_dict['template']['probe']
        templates_params['n'] = target_template_dict['template']['N']
        templates_params['seed'] = 671

        if target_template_dict['template']['drifting'] != 'None':
            templates_params["drifting"] = True
            templates_params["drift_steps"] = 30
            templates_params["drift_xlim"] = [-5, 5]
            templates_params["drift_ylim"] = [-5, 5]
            templates_params["drift_zlim"] = [100, 100]
            templates_params["max_drift"] = 200

        tempgen = mr.gen_templates(cell_models_folder=cell_models_folder, params=templates_params, verbose=True, n_jobs=32)
        mr.save_template_generator(tempgen=tempgen, filename=f'{dataset_folder}/{len(yaml_files)+1}.h5')
        with open(f'{dataset_folder}/{len(yaml_files)+1}.yaml', 'w') as yaml_file:
            yaml.dump(target_template_dict, yaml_file)
        template_file = f'{dataset_folder}/{len(yaml_files)+1}.h5'

    return template_file


def train(model: nn.Module,
          train_dataset: Dataset,
          test_dataset: Dataset,
          bs: int,
          lr: float,
          epoch: int,
          verbose: False):
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0
    )

    optimizer = optim.Adam(
        #filter(lambda p: p.requires_grad==True, model.parameters()),
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999)
    )
    # Configure criterion
    criterion = nn.BCELoss()
    
    # Main training loop
    all_train_loss = np.zeros(epoch)
    all_acc_train = np.zeros(epoch)
    all_acc_test = np.zeros(epoch)
    maxAcc = 0.0
    num_epochs = epoch
    if not verbose:
        t_bar = tqdm(total=num_epochs)
    for e in range(num_epochs):
        # Train for this epoch
        model.train()
        accLoss = 0.0
        correct = 0
        wrong = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            target = target.to(torch.float)
            loss = criterion(output, target)
            pred = (output >= 0.5).float()
            correct += ((pred+target) == 2).float().sum()
            wrong += ((pred+target) == 1).float().sum()
            accLoss += loss.detach() * len(data)
            loss.backward()
            optimizer.step()

        accLoss /= len(train_loader.dataset)
        accuracy = 100 * correct / (correct+wrong)
        val_accuracy = test(model, train_dataset)
        test_accuracy = test(model, test_dataset)
        all_train_loss[e] = accLoss
        all_acc_train[e] = val_accuracy
        all_acc_test[e] = test_accuracy
        if not verbose:
            t_bar.update(1)
            t_bar.set_description(f'loss/train/val/test accuracy: {accLoss} {accuracy:.2f}% {val_accuracy:.2f}% {test_accuracy:.2f}%')
            t_bar.refresh
        else:
            print(f'[{e}] loss/train/val/test accuracy: {accLoss} {accuracy:.2f}% {val_accuracy:.2f}% {test_accuracy:.2f}%')
    return all_acc_train, all_acc_test


def test(model, dataset):
    model.eval()
    correct = 0
    wrong = 0
    with torch.no_grad():
        for i in range(np.ceil((dataset.recording.shape[0]-60)/10000).astype(np.int32)):
            selected_slices = dataset.recording[10000*i:10000*(i+1)+60-1].unfold(dimension=0, size=60, step=1).permute([0,2,1])
            output = model(selected_slices)
            pred = (output >= 0.5).float()
            correct += ((pred+dataset.spiketrains[10000*i:10000*(i+1)]) == 2).float().sum()
            wrong += ((pred+dataset.spiketrains[10000*i:10000*(i+1)]) == 1).float().sum()
    if (float(correct)+float(wrong) != 0):
        accuracy = 100 * float(correct) / (float(correct)+float(wrong))
    else:
        accuracy = 0
    return accuracy

def eval(model, dataset):
    eval_result = np.zeros([25,4])
    for t in range(25):
        th = 0.5 + t * 0.1
        model.eval()
        num_points = 17
        triangle_kernel = torch.cat([torch.linspace(0,1,(num_points+3)//2)[:-1], torch.linspace(1,0,(num_points+3)//2)])[1:-1]
        triangle_kernel = triangle_kernel[None][None].to("cuda")
        spikes = torch.Tensor().to("cuda")
        with torch.no_grad():
            for i in range(np.ceil((dataset.recording.shape[0]-60)/10000).astype(np.int32)):
                selected_slices = dataset.recording[10000*i:10000*(i+1)+60-1].unfold(dimension=0, size=60, step=1).permute([0,2,1])
                pred = model(selected_slices)
                pred = torch.nn.functional.conv1d(input=pred[None].permute([2,0,1]), weight=triangle_kernel)[:,0,:]
                mid = pred[:,1:-1] >= th
                left = (pred[:,1:-1] - pred[:,0:-2]) >= 0
                right = (pred[:,1:-1] - pred[:,2:]) >= 0
                indices = (mid*left*right).nonzero(as_tuple=True)
                spikes_sub = torch.cat([pred[indices[0], indices[1]+1][None] ,indices[0][None], indices[1][None]+10000*i+16]).permute([1, 0])
                spikes = torch.cat([spikes, spikes_sub])
        tp_amp = torch.Tensor([]).to("cuda")
        fp_amp = torch.Tensor([]).to("cuda")
        n_fn = 0
        for neuron in tqdm(range(dataset.spiketrains.shape[1])):
            gt_spikes = dataset.spiketrains[:, neuron].nonzero().reshape(-1)
            pred_spikes = spikes[spikes[:,1] == neuron]
            for gt_spike in gt_spikes:
                match = pred_spikes[torch.abs(pred_spikes[:,2] - gt_spike) < 30]
                if (match.shape[0] != 0):
                    index = torch.argmin(torch.abs(pred_spikes[:,2] - gt_spike))
                    tp_amp = torch.cat([tp_amp, pred_spikes[index, 0][None]])
                    pred_spikes = torch.cat([pred_spikes[0:index], pred_spikes[index+1:]], dim=0)
                else:
                    n_fn += 1
            fp_amp = torch.cat([fp_amp, pred_spikes[:, 0]])
        eval_result[t][0] = tp_amp.shape[0]
        eval_result[t][1] = fp_amp.shape[0]
        eval_result[t][2] = n_fn
        eval_result[t][3] = tp_amp.shape[0] / (tp_amp.shape[0] + fp_amp.shape[0] + n_fn)
    return eval_result

def eval_result(model_index, th=0.5):
    model = torch.load(f'../run/{model_index}.pt')
    with open(f'../run/{model_index}.yaml', 'r') as yaml_file:
        target_yaml = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    recording_dict = {
        'recording': target_yaml['finetune']['recording']
    }
    model.eval()
    recording_file = create_recording(recording_dict, '../rec', '../config')
    validset = MEARecDataset_full(recording_file=recording_file,
                                  start=target_yaml['finetune']['recording']['duration'] // 2,
                                  end=target_yaml['finetune']['recording']['duration'],
                                  jitter=0)
    start_time = time.time()
    num_points = 17
    triangle_kernel = torch.cat([torch.linspace(0,1,(num_points+3)//2)[:-1], torch.linspace(1,0,(num_points+3)//2)])[1:-1]
    triangle_kernel = triangle_kernel[None][None].to("cuda")
    spikes = torch.Tensor().to("cuda")
    with torch.no_grad():
        for i in range(np.ceil((validset.recording.shape[0]-60)/10000).astype(np.int32)):
            selected_slices = validset.recording[10000*i:10000*(i+1)+60-1].unfold(dimension=0, size=60, step=1).permute([0,2,1])
            pred = model(selected_slices)
            pred = torch.nn.functional.conv1d(input=pred[None].permute([2,0,1]), weight=triangle_kernel)[:,0,:]
            mid = pred[:,1:-1] >= th
            left = (pred[:,1:-1] - pred[:,0:-2]) >= 0
            right = (pred[:,1:-1] - pred[:,2:]) >= 0
            indices = (mid*left*right).nonzero(as_tuple=True)
            spikes_sub = torch.cat([pred[indices[0], indices[1]+1][None] ,indices[0][None], indices[1][None]+10000*i+16]).permute([1, 0])
            spikes = torch.cat([spikes, spikes_sub])
        etc_time = time.time() - start_time
        tp_amp = torch.Tensor([]).to("cuda")
        fp_amp = torch.Tensor([]).to("cuda")
        n_fn = 0
        for neuron in tqdm(range(validset.spiketrains.shape[1])):
            gt_spikes = validset.spiketrains[:, neuron].nonzero().reshape(-1)
            pred_spikes = spikes[spikes[:,1] == neuron]
            for gt_spike in gt_spikes:
                match = pred_spikes[torch.abs(pred_spikes[:,2] - gt_spike) < 30]
                if (match.shape[0] != 0):
                    index = torch.argmin(torch.abs(pred_spikes[:,2] - gt_spike))
                    tp_amp = torch.cat([tp_amp, pred_spikes[index, 0][None]])
                    pred_spikes = torch.cat([pred_spikes[0:index], pred_spikes[index+1:]], dim=0)
                else:
                    n_fn += 1
            fp_amp = torch.cat([fp_amp, pred_spikes[:, 0]])
    accuracy = tp_amp.shape[0] / (tp_amp.shape[0] + fp_amp.shape[0] + n_fn)
    return accuracy, etc_time

def run_sorter(recording_index, sorter='kilosort4'):
    recording_file = f'../rec/{recording_index}.h5'
    recording = se.MEArecRecordingExtractor(recording_file)
    sorting_GT = se.MEArecSortingExtractor(recording_file)
    sorting = ss.run_sorter(sorter_name=sorter, recording=recording, remove_existing_folder=True, delete_output_folder=True, delete_container_files=True)
    comp_KS4 = sc.compare_sorter_to_ground_truth(sorting_GT, sorting)
    num_tp = comp_KS4.count_score.tp.sum()
    num_fp = comp_KS4.count_score.fp.sum()
    num_fn = comp_KS4.count_score.fn.sum()
    accuracy = num_tp / (num_tp + num_fp + num_fn)
    
    recording_slice = recording.time_slice(50,100)
    start_time = time.time()
    sorting = ss.run_sorter(sorter_name=sorter, recording=recording_slice, remove_existing_folder=True, delete_output_folder=True, delete_container_files=True)
    etc_time = time.time() - start_time
    return accuracy, etc_time
