from sklearn.base import BaseEstimator
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn

num_epochs = 20
BATCH = 50
NB_GAUSSIANS = 2
EPSILON = 1e-12

# In this submission, we try to estimate the true distribution by using
# NB_GAUSSIANS gaussian distributions, with mu and sigma's estimated
# from a pytorch model, and an additional uniform distribution that
# covers the rest of the space (avoiding to have the likelihood drop to 0
# in the case where no distribution covers the true value)

def loglk(x, mean, sd):
    var = torch.mul(sd, sd)
    denom = torch.sqrt(2 * np.pi * var)
    diff = x - mean
    num = torch.exp(-torch.mul(diff, diff) / (2 * var))
    probs = num / denom
    return probs

# Custom pytorch loss, giving an approximate of the log likelihood
# when made of gaussians
class CustomLoss:
    def __init__(self, weights):
        self.weights = torch.Tensor(weights[1:])

    def __call__(self, y_pred, y_true):
        mus, sigmas = y_pred[:len(y_true), ], y_pred[len(y_true):, ]

        probs = loglk(y_true, mus, sigmas)

        summed_prob = torch.sum(probs * self.weights, dim=1)

        summed_prob = torch.clamp(summed_prob, EPSILON)

        log_lk = -torch.log(summed_prob)

        log_lk = torch.sum(log_lk) / (mus.shape[0] * mus.shape[1] * y_true.shape[1])

        if log_lk != log_lk:
            raise ValueError("Nan in loss")
        return log_lk


def train_model(model, dataset, optimizer, n_epochs=10, batch_size=128,
                loss_fn=None):
    r""" Training model over dataset with loss function loss_fn.
    """
    model.train()

    dataset = data.DataLoader(dataset, batch_size=batch_size)

    for epoch in range(n_epochs):

        for (idx, (x, y)) in enumerate(dataset):
            x, y = Variable(x), Variable(y)
            model.zero_grad()

            out = model(x)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    return model


def loss_model(model, dataset, batch_size=None, loss_fn=nn.MSELoss()):
    r""" Tests the model over a dataset with loss function loss_fn and prints
    the result.
    """

    if batch_size is None:
        batch_size = len(dataset)

    model.eval()

    dataset = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # we only take the first batch and return its loss
    with torch.no_grad():
        (x, y) = next(iter(dataset))
        out = model(x)
        loss = loss_fn(out, y)
        print('Test Loss: {:.4f}'.format(loss.item()))
    return loss


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, curr_fold):
        self.max_dists = max_dists
        self.a = None
        self.b = None

        weights = (np.arange(NB_GAUSSIANS + 1) + 1)
        weights = weights.astype(float)
        weights[0] = EPSILON
        self.weights = weights / sum(weights)

    def fit(self, X, y):
        self.a = np.min(y)
        self.b = np.max(y)
        self.model = SimpleBinnedNoBounds(NB_GAUSSIANS, X.shape[1])
        dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
        optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

        loss = CustomLoss(self.weights)

        train_model(self.model, dataset, optimizer=optimizer,
                    n_epochs=num_epochs, batch_size=BATCH,
                    loss_fn=loss)
        loss_model(self.model, dataset, loss_fn=loss, batch_size=BATCH)

    def predict(self, X):
        self.model.eval()
        X = torch.Tensor(X)
        batch = BATCH
        num_data = X.shape[0]
        num_batches = int(num_data / batch) + 1
        all_res_sigma = []
        all_res_mu = []
        for i in range(num_batches):
            size = min((i + 1) * batch, num_data) - i * batch
            y_pred = \
                self.model(X[i * batch:min((i + 1) * batch, num_data)])

            batch_res_mu, batch_res_sigma = y_pred[:size, ], y_pred[size:, ]

            all_res_sigma.append(
                (batch_res_sigma.detach().numpy())
            )
            all_res_mu.append(batch_res_mu.detach().numpy())

        mus = np.concatenate(all_res_mu)
        sigmas = np.concatenate(all_res_sigma)

        weights = np.stack([self.weights
                            for _ in range(len(mus))],
                           axis=0)

        # We put each mu next to its sigma
        params_normal = np.empty((len(X), NB_GAUSSIANS * 2))
        params_normal[:, 0::2] = mus
        params_normal[:, 1::2] = sigmas

        # Uniform
        a_array = np.array([self.a] * len(X))
        b_array = np.array([self.b] * len(X))
        params_uniform = np.stack((a_array, b_array), axis=1)

        # We concatenate the params
        # To get information about the parameters of the distribution you are
        # using, you can run
        #   import rampwf as rw
        #   [(v,v.params) for v in rw.utils.distributions_dict.values()]
        params = np.concatenate((params_uniform, params_normal), axis=1)

        # The first generative regressors is uniform, the others are gaussians
        # For the whole list of distributions, run
        #   import rampwf as rw
        #   rw.utils.distributions_dict
        types = np.zeros(NB_GAUSSIANS + 1)
        types[0] = 1
        types = np.array([types] * len(X))

        return weights, types, params


class SimpleBinnedNoBounds(nn.Module):
    def __init__(self, nb_sigmas, input_size):
        super(SimpleBinnedNoBounds, self).__init__()

        output_size_sigma = nb_sigmas
        output_size_mus = nb_sigmas
        layer_size = 50

        self.linear = nn.Linear(input_size, layer_size)
        self.act = nn.LeakyReLU()

        self.linear_mus = nn.Linear(layer_size, layer_size)
        self.act_mus = torch.nn.LeakyReLU()
        self.linear_mus_2 = nn.Linear(layer_size, output_size_mus)

        self.linear_sigma = nn.Linear(layer_size + output_size_mus, layer_size)
        self.act_sigma = torch.nn.LeakyReLU()
        self.linear_sigma_2 = nn.Linear(layer_size, output_size_sigma)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)

        mu = self.linear_mus(x)
        mu = self.act_mus(mu)
        mu = self.linear_mus_2(mu)

        x = torch.cat((x, mu), dim=1)

        sigma = self.linear_sigma(x)
        sigma = self.act_sigma(sigma)
        sigma = self.linear_sigma_2(sigma)
        sigma = torch.exp(sigma)
        return torch.cat([mu, sigma], dim=0)
