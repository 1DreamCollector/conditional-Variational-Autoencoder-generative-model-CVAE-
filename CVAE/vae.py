import  torch
from    torch import nn





class VAE(nn.Module):



    def __init__(self,num_class):
        super(VAE, self).__init__()


        # [b, 4096] => [b, 20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.encoder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # [b, 20] => [b, 4096]
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 4096),
            nn.Sigmoid()
        )
        self.dense = nn.Sequential(nn.Embedding(num_class,50)
                                 ,nn.Linear(50,10)
                                 )
        self.criteon = nn.MSELoss()

    def forward(self, x, labels):
        """

        :param x: [b, 1, 64, 64]
        :return:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, 4096)
        # encoder
        # [b, 20], including mean and sigma
        h_ = self.encoder(x)
        # [b, 20] => [b, 10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        mu_y = self.dense(labels)
        # reparametrize trick, epison~N(0, 1)
        h = mu + sigma * torch.randn_like(sigma)

        # decoder
        x_hat = self.decoder(h)
        # reshape
        x_hat = x_hat.view(batchsz, 1, 64, 64)

        kld = 0.5 * torch.sum(
            torch.pow((mu-mu_y), 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz*64*64)

        return x_hat, kld