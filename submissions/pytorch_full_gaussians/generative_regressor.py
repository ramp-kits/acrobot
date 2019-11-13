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
# NB_GAUSSIANS gaussian distributions and a uniform.

# This pythorch submissions does not use a validation dataset, or any
# sophisticated techniques. It is to be refined an improved. When done
# correctly, it can outperform significantly other methods


class GenerativeRegressor(BaseEstimator):
    def __init__(self, max_dists, target_dim):
        """
        Parameters
        ----------
        max_dists : int
            The maximum number of distributions (kernels) in the mixture.
        target_dim : int
            The index of the target column to be predicted.
        """
        self.max_dists = max_dists
        self.a = None
        self.b = None

        # The weights are gradually increasing, with
        weights = (np.arange(NB_GAUSSIANS + 1) + 1)
        weights = weights.astype(float)
        # The first distribution (uniform one) will have EPSILON as weight
        weights[0] = EPSILON
        # In this setup, weights are approximately :
        # [2e-13, 0.399, 0.599]
        self.weights = weights / sum(weights)

    def fit(self, X, y):
        """
        Mus and sigmas pairs are estimated from a pytorch model
        """
        # We create a model, and give it the number of gaussians desired
        # along with the input shape
        self.model = SimpleBinnedNoBounds(NB_GAUSSIANS, X.shape[1])

        # We create a dataset
        dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))

        # We initialise a simple optimiser
        optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

        # We create our custom loss (an approxitame of the negative log
        # likelihood that considers only the gaussians). This has to
        # adapt to what your network outputs, and should have smooth gradiants
        # to make training easier.
        loss = CustomLoss(self.weights)

        # Similar to reg.fit(X, y)
        train_model(self.model, dataset, optimizer=optimizer,
                    n_epochs=num_epochs, batch_size=BATCH,
                    loss_fn=loss)

        # We evaluate how good we did with respect to our custom loss
        loss_model(self.model, dataset, loss_fn=loss, batch_size=BATCH)

        # We find a single a and a single b to cover the space of our dimension
        # In this example, it is just min and max
        self.a = np.min(y)
        self.b = np.max(y)

    def predict(self, X):
        """Construct a conditional mixture distribution.
        Return
        ------
        weights : np.array of float
            discrete probabilities of each component of the mixture
        types : np.array of int
            integer codes referring to component types
            see rampwf.utils.distributions_dict
        params : np.array of float tuples
            parameters for each component in the mixture
        """
        self.model.eval()
        X = torch.Tensor(X)
        batch = BATCH
        num_data = X.shape[0]
        num_batches = int(num_data / batch) + 1
        # For every batch of data, we get predictions for mus and sigmas
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
        # we get two t by NB_GAUSSIANS matrices

        weights = np.stack([self.weights
                            for _ in range(len(mus))],
                           axis=0)

        # We put each mu next to its sigma
        params_normal = np.empty((len(X), NB_GAUSSIANS * 2))
        params_normal[:, 0::2] = mus
        params_normal[:, 1::2] = sigmas
        # Now params_normal is a t by 2*NB_GAUSSIANS matrix

        # We set the uniform distributions parameters in the same way
        a_array = np.array([self.a] * len(X))
        b_array = np.array([self.b] * len(X))
        params_uniform = np.stack((a_array, b_array), axis=1)

        # We concatenate the params
        params = np.concatenate((params_uniform, params_normal), axis=1)

        # To get information about the parameters of the distribution you are
        # using, you can run
        #   import rampwf as rw
        #   [(v,v.params) for v in rw.utils.distributions_dict.values()]

        # The first generative regressors is uniform, the others are gaussians
        types = np.zeros(NB_GAUSSIANS + 1)
        types[0] = 1
        types = np.array([types] * len(X))
        # For the whole list of distributions, run
        #   import rampwf as rw
        #   rw.utils.distributions_dict

        return weights, types, params


class SimpleBinnedNoBounds(nn.Module):
    """
    A simple neural net with the following shape :
                     +---+
          +---+  +-> |   |-> mus
          |   |  |   +---+
     X -> |   +--+     |
          |   |  |   +---+
          +---+  +-> |   |-> sigmas
                     +---+

    For each time step, we output nb_sigmas pairs of mu,sigma
    """

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


def pdf_norm(x, mean, sd):
    """ Pdf for gaussains in pytorch"""
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

        probs = pdf_norm(y_true, mus, sigmas)

        summed_prob = torch.sum(probs * self.weights, dim=1)

        # This clamping sets the minimal log likelihood, preventing from
        # having too bad of a score if none of the predictions is right.
        # This is not present in the tru log likelihood, and is only used
        # to make training smoother.
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
        print('Loss: {:.4f}'.format(loss.item()))
    return loss
