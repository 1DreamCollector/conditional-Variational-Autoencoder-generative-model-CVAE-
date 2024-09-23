import  torch
from    torch.utils.data import DataLoader
from    torch import nn, optim
from data_process.CWRU_WP_augmentation import WP_aug
from    vae import VAE
import  visdom
import matplotlib.pyplot as plt

def main():
    num_class = 10
    dataset_name = "CWRU"
    CWRU_aug = WP_aug()
    dataset_train,_,_,fault_name = CWRU_aug.WP_aug(datadir="D:\BaiduNetdiskDownload\CRWU凯斯西储大学轴承数据\CRWU")

    dataloader_train = DataLoader(dataset_train, batch_size=600, shuffle=False,drop_last=False)

    # mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)
    x, _ = iter(dataloader_train).next()
    print('x:', x.shape)

    device = torch.device('cuda')
    # model = AE().to(device)
    model = VAE(num_class=num_class).to(device)
    model.load_state_dict(torch.load("D:\edge\FLOW\CVAE\save_model\CWRU\CVAE_dataset_CWRU_epoch_999_loss_0.6837758421897888_kld_0.0016708346083760262.pt"))
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    # viz = visdom.Visdom()

    for epoch in range(1000):
        # for batchidx, (x, labels) in enumerate(dataloader_train):
        #     # [b, 1, 28, 28]
        #     x = x.to(device)
        #     labels = labels.to(device)
        #     x_hat, kld = model(x, labels)
        #     loss = criteon(x_hat, x)
        #
        #     if kld is not None:
        #         elbo = - loss - 1.0 * kld
        #         loss = - elbo
        #
        #     # backprop
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #
        # print(epoch, 'loss:', loss.item(), 'kld:', kld.item())
        # if(epoch %999==0):
        #     torch.save(model.state_dict(), f"save_model/{dataset_name}/CVAE_dataset_{dataset_name}_epoch_{epoch}_loss_{loss}_kld_{kld}.pt")

        with torch.no_grad():

            last_img = None
            for i in range(num_class):
                label = torch.ones(5).to(device)
                label = (label*i).long()
                mu = model.dense(label)
                z = (mu + torch.randn_like(mu)).to(device)
                img = model.decoder(z)
                img = img.view(-1, 1, 64, 64)
                img = img.squeeze().cpu().numpy()

                for index,img in enumerate(img):
                    fig = plt.figure(figsize=(10, 10))
                    ax1 = plt.subplot(1, 1, 1)
                    img = ax1.imshow(img, interpolation="bilinear", cmap="plasma", aspect="auto")
                    ax1.set_title(f'{fault_name[i]}', fontsize=11)
                    fig.colorbar(img, ax=ax1)
                    plt.savefig(
                        f'sample/{dataset_name}/epoch_{epoch}_class_{fault_name[i]}_index{index}.png',
                        bbox_inches='tight', dpi=50)

                    plt.close()

            labels = [i for i in range(10)]
            fixed_label=[]
            for i in labels:
                label = (torch.ones(100)*i).to(device)
                fixed_label.append(label)
            fixed_label = torch.stack(fixed_label).long().to(device)
            mu = model.dense(fixed_label)
            z = (mu + torch.randn_like(mu)).to(device)
            img = model.decoder(z)
            img = img.view(-1, 1, 64, 64).squeeze().cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            X = img.reshape(100 * 10, -1)
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)
            X_embedded = tsne.fit_transform(X)
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'brown', 'orange', 'purple']  # 为每个类别指定颜色
            for i in range(10):  # 假设有5个类别
                ax.scatter(X_embedded[i * 100:(i + 1) * 100, 0], X_embedded[i * 100:(i + 1) * 100, 1],
                           c=colors[i],
                           label=fault_name[i], s=10)

            legend = ax.legend(loc='best', fontsize='medium')
            plt.tight_layout()
            plt.savefig(r'D:\edge\FLOW\CVAE\CWRUvisualization.png', dpi=300)



if __name__ == '__main__':
    main()