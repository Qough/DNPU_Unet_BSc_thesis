import torch
import lightning
import numpy
import librosa
import os
import sounddevice
import random
from glob import glob


class U_netDown(torch.nn.Module):
    def __init__(self):
        super(U_netDown, self).__init__()
        self.convLayer1 = torch.nn.Conv1d(1, 64, 3, padding='same')
        self.convLayer2 = torch.nn.Conv1d(64, 64, 3, padding='same')

        self.poolLayer3 = torch.nn.MaxPool1d(2)

        self.convLayer4 = torch.nn.Conv1d(64, 128, 3, padding='same')
        self.convLayer5 = torch.nn.Conv1d(128, 128, 3, padding='same')

        self.poolLayer6 = torch.nn.MaxPool1d(2)

        self.convLayer7 = torch.nn.Conv1d(128, 256, 3, padding='same')
        self.convLayer8 = torch.nn.Conv1d(256, 256, 3, padding='same')

        self.poolLayer9 = torch.nn.MaxPool1d(2)

        self.convLayer10 = torch.nn.Conv1d(256, 512, 3, padding='same')
        self.convLayer11 = torch.nn.Conv1d(512, 512, 3, padding='same')

        self.poolLayer12 = torch.nn.MaxPool1d(2)

        self.convLayer13 = torch.nn.Conv1d(512, 1028, 3, padding='same')
        self.convLayer14 = torch.nn.Conv1d(1028, 1028, 3, padding='same')


    def forward(self, x):

        x = self.convLayer1(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer2(x)
        x = torch.nn.ReLU()(x)

        x = self.poolLayer3(x)

        x = self.convLayer4(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer5(x)
        x = torch.nn.ReLU()(x)

        x = self.poolLayer6(x)

        x = self.convLayer7(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer8(x)
        x = torch.nn.ReLU()(x)

        x = self.poolLayer9(x)

        x = self.convLayer10(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer11(x)
        x = torch.nn.ReLU()(x)

        x = self.poolLayer12(x)

        x = self.convLayer13(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer14(x)
        x = torch.nn.ReLU()(x)

        return x


class U_netUp(torch.nn.Module):
    def __init__(self):
        super(U_netUp, self).__init__()
        self.transLayer15 = torch.nn.ConvTranspose1d(1028, 1028, 2, stride=2)

        self.convLayer16 = torch.nn.Conv1d(1028, 512, 3, padding='same')
        self.convLayer17 = torch.nn.Conv1d(512, 512, 3, padding='same')

        self.transLayer18 = torch.nn.ConvTranspose1d(512, 512, 2, stride=2)

        self.convLayer19 = torch.nn.Conv1d(512, 256, 3, padding='same')
        self.convLayer20 = torch.nn.Conv1d(256, 256, 3, padding='same')

        self.transLayer21 = torch.nn.ConvTranspose1d(256, 256, 2, stride=2)

        self.convLayer22 = torch.nn.Conv1d(256, 128, 3, padding='same')
        self.convLayer23 = torch.nn.Conv1d(128, 128, 3, padding='same')

        self.transLayer24 = torch.nn.ConvTranspose1d(128, 128, 2, stride=2)

        self.convLayer25 = torch.nn.Conv1d(128, 64, 3, padding='same')
        self.convLayer26 = torch.nn.Conv1d(64, 64, 3, padding='same')

        # Single channel output or duel channel output at written in the U net paper
        # self.convLayer27 = torch.nn.Conv1d(64, 2, 1)
        self.convLayer27 = torch.nn.Conv1d(64, 1, 1, padding='same')

    def forward(self, x):
        x = self.transLayer15(x)

        x = self.convLayer16(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer17(x)
        x = torch.nn.ReLU()(x)

        x = self.transLayer18(x)

        x = self.convLayer19(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer20(x)
        x = torch.nn.ReLU()(x)

        x = self.transLayer21(x)

        x = self.convLayer22(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer23(x)
        x = torch.nn.ReLU()(x)

        x = self.transLayer24(x)

        x = self.convLayer25(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer26(x)
        x = torch.nn.ReLU()(x)
        x = self.convLayer27(x)

        return x


class U_netEncoder(lightning.LightningModule):

    def __init__(self, net):
        super(U_netEncoder, self).__init__()
        self.net = net

    def training_step(self, batch, batch_idx):
        noisyAudio, cleanAudio = batch
        #noisyAudio = noisyAudio.view(noisyAudio.size(0), -1)
        denoisedAudio = self.net(noisyAudio)

        MSELoss = torch.nn.MSELoss()(denoisedAudio, cleanAudio)

        #sounddevice.play(denoisedAudio.squeeze().detach().cpu().numpy(), 8000)

        self.log("train_loss", MSELoss, prog_bar=True)

        return MSELoss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        # return torch.optim.AdamW(self.parameters(), lr=1e-3)


class U_netDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 audioDirectory : str,
                 noiseDirectory : str = None,
                 SNRdB: int = 10,
                 sampleFrequency = None):

        # Fetches every .wav file in the root directory
        self.audioFiles = glob(audioDirectory + "/*.wav")

        self.SNRdB = SNRdB

        self.sampleFrequency = sampleFrequency

        if noiseDirectory:
            self.addNoise = self.addCustomNoise
            self.noiseFiles = glob(noiseDirectory + "/*.wav")
        else:
            self.addNoise = self.addGausianNoise
            self.noiseFiles = None

    def __len__(self):
        return len(self.audioFiles)
   
    def addCustomNoise(self, cleanAudio, audioSampleFrequency):
        # Calculates the RMS
        SNRLinear = 10**(self.SNRdB / 10)
        RMSAudio = numpy.sqrt(numpy.mean(cleanAudio ** 2))
        wantedRMSNoise = RMSAudio / numpy.sqrt(SNRLinear)

        # Loads in the noise file at the same sample rate 
        noise, _ = librosa.load(random.choice(self.noiseFiles), sr = audioSampleFrequency)

        # Calculates the RMS of the noise file
        RMSNoise = numpy.sqrt(numpy.mean(noise ** 2))

        # Scales the noise to achieve the desire SNR
        noise = noise * (wantedRMSNoise / RMSNoise)

        # Tiles the noise if the noise file is shorter then the audio file
        # So its length is always equel or greater than the length of the audio file
        if len(noise) < len(cleanAudio):
            sizeDifference = int(numpy.ceil(len(cleanAudio) / len(noise)))
            noise = numpy.tile(noise, sizeDifference)

        return cleanAudio + noise[:len(cleanAudio)]

    def addGausianNoise(self, cleanAudio, _):
        SNRLinear = 10**(self.SNRdB/10)
        RMSAudio = numpy.sqrt(numpy.mean(cleanAudio**2))
        RMSNoise = RMSAudio / numpy.sqrt(SNRLinear)

        noise = numpy.random.normal(0, RMSNoise, cleanAudio.shape)

        return cleanAudio + noise

    def __getitem__(self, idx):
        cleanAudio, audioSampleFrequency = librosa.load(self.audioFiles[idx], sr = self.sampleFrequency)

        cleanAudio = numpy.float32(numpy.pad(cleanAudio, (0, -1 * (len(cleanAudio) % -16))))

        noisyAudio = self.addNoise(cleanAudio, audioSampleFrequency)
        
        #label = os.path.basename(self.audioFiles[idx])[:1]

        return noisyAudio, cleanAudio