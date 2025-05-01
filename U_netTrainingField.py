import U_net
import torch
import lightning


def main():

    down = U_net.U_netDown()
    up = U_net.U_netUp()
    net = torch.nn.Sequential(down, up)

    learnableParameterNo = sum(p.numel() for p in net.parameters())
    print(learnableParameterNo)

    encoder = U_net.U_netEncoder(net)

    pathToAudio = "D:\\UTwente\\Mod 12\\Code\\free-spoken-digit-dataset-master\\recordings_zero"
    pathToNoise = "D:\\UTwente\\Mod 12\\Code\\UrbanSound8K\\audio\\9_street_music"

    dataset = U_net.U_netDataset(pathToAudio, pathToNoise)

    # print(len(dataset))
    # print(dataset[0])

    train_loader = torch.utils.data.DataLoader(dataset)

    trainer = lightning.Trainer(max_epochs=5)
    trainer.fit(model=encoder, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
