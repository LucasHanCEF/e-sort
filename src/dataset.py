import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import MEArec as mr
import MEAutility as MEA

class MEARecDataset_full(Dataset):
    def __init__(self,
                 recording_file: str,
                 start: int,
                 end: int,
                 jitter: int
                 ):
        super(MEARecDataset_full, self).__init__()
        self.recgen = mr.load_recordings(recording_file)
        self.recording = torch.from_numpy(self.recgen.recordings[int(start*32e3):int(end*32e3),:]).to("cuda")
        self.spiketrains = torch.zeros([self.recgen.recordings.shape[0], len(self.recgen.spiketrains)], dtype=torch.bool)
        for (neuron_index, spiketrain) in enumerate(self.recgen.spiketrains):
            for spike in spiketrain:
                center_time = int(np.round(spike.magnitude/0.03125e-3))
                if center_time >= 30-jitter:
                    self.spiketrains[center_time-jitter:center_time+1+jitter,neuron_index] = True
        self.spiketrains = self.spiketrains[int(start*32e3+29):int(end*32e3-30),:].to("cuda")
    
    def __len__(self):
        return self.recording.shape[0]-60
    
    def __getitem__(self, index):
        return self.recording[index:index+60], self.spiketrains[index]


class MEARecDataset_window(Dataset):
    def __init__(self,
                 recording_file: str,
                 start: int,
                 end: int,
                 jitter: int
                 ):
        super(MEARecDataset_window, self).__init__()
        recgen = mr.load_recordings(recording_file)
        self.recording = torch.from_numpy(recgen.recordings[int(start*32e3):int(end*32e3),:]).to("cuda")
        spiketrains = torch.zeros([recgen.recordings.shape[0], len(recgen.spiketrains)], dtype=torch.bool).to("cuda")
        for (neuron_index, spiketrain) in enumerate(recgen.spiketrains):
            for spike in spiketrain:
                center_time = int(np.round(spike.magnitude/0.03125e-3))
                if center_time >= 30-jitter:
                    spiketrains[center_time-jitter:center_time+1+jitter,neuron_index] = True
        self.spiketrains = spiketrains[int(start*32e3+29):int(end*32e3-30),:]
        spike_indices = torch.sum(self.spiketrains, dim=1).nonzero()

        noise_indices = (torch.sum(self.spiketrains, dim=1) == 0).nonzero()
        noise_indices = noise_indices[torch.randperm(noise_indices.shape[0])[:spike_indices.shape[0]]]

        self.windows = []
        self.labels = []
        for index in spike_indices:
            self.windows.append(self.recording[index:index+60,:])
            self.labels.append(self.spiketrains[index,:][0])
        for index in noise_indices:
            self.windows.append(self.recording[index:index+60,:])
            self.labels.append(self.spiketrains[index,:][0])

    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        return self.windows[index], self.labels[index]

class MEARecDataset_window_number(Dataset):
    def __init__(self,
                 recording_file: str,
                 window_number: int,
                 jitter: int
                 ):
        super(MEARecDataset_window_number, self).__init__()
        recgen = mr.load_recordings(recording_file)
        max_time = 0
        for (neuron_index, spiketrain) in enumerate(recgen.spiketrains):
            max_time_spiketrain = int(np.round(spiketrain[window_number-1].magnitude/0.03125e-3))
            if max_time_spiketrain > max_time:
                max_time = max_time_spiketrain
        self.recording = torch.from_numpy(recgen.recordings[0:max_time+60,:]).to("cuda")
        spiketrains = torch.zeros([recgen.recordings.shape[0], len(recgen.spiketrains)], dtype=torch.bool).to("cuda")
        for (neuron_index, spiketrain) in enumerate(recgen.spiketrains):
            for spike in spiketrain:
                center_time = int(np.round(spike.magnitude/0.03125e-3))
                if center_time >= 30-jitter:
                    spiketrains[center_time-jitter:center_time+1+jitter,neuron_index] = True
        self.spiketrains = spiketrains[29:max_time+30,:]
        
        spike_indices = torch.Tensor().to("cuda")
        for neuron_index in range(self.spiketrains.shape[1]):
            spike_index = self.spiketrains[:,neuron_index].nonzero()[:window_number*(2*jitter+1)]
            spike_indices = torch.cat([spike_indices, spike_index])
        spike_indices = spike_indices.reshape(-1)

        noise_indices = (torch.sum(self.spiketrains, dim=1) == 0).nonzero()
        noise_indices = noise_indices[torch.randperm(noise_indices.shape[0])[:spike_indices.shape[0]]]

        self.windows = []
        self.labels = []
        for index in spike_indices:
            index = int(index)
            self.windows.append(self.recording[index:index+60,:])
            self.labels.append(self.spiketrains[index,:])
        for index in noise_indices:
            index = int(index)
            self.windows.append(self.recording[index:index+60,:])
            self.labels.append(self.spiketrains[index,:])

    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        return self.windows[index], self.labels[index]
