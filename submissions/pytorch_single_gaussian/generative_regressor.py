from sklearn.base import BaseEstimator
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn

num_epochs = 20
batch = 200

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
        self.sigma = None
        self.a = None
        self.b = None

    def fit(self, X, y):
        """
        Pytorch simple regressor to estimate the mu of a# gaussian,
        we estimate the sigma over the training set.
        Additionally, we use a uniform distribution that covers the rest of
        the space (avoiding to have the likelihood drop to 0 in the case where
         no distribution covers the true value)
        """
        self.model = PytorchReg(X.shape[1])

        # We create a dataset
        dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))

        # We initialise an optimiser
        optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

        # Similar to reg.fit(X, y)
        train_model(self.model, dataset, optimizer=optimizer,
                    n_epochs=num_epochs, batch_size=batch,
                    loss_fn=nn.MSELoss())

        # We evaluate how good we did with respect to the MSE
        loss_model(self.model, dataset, loss_fn=nn.MSELoss(), batch_size=batch)

        self.model.eval()
        X = torch.Tensor(X)

        # We run our model over the whole training data to get an estimate
        # of sigma (in a batched fashion)
        num_data = X.shape[0]
        num_batches = int(num_data / batch)
        yGuess = []
        for i in range(num_batches):
            batch_res = \
                self.model(X[i * batch:min((i + 1) * batch, num_data)])
            yGuess += list(batch_res.detach().cpu().numpy().ravel())

        yGuess = np.array(yGuess).reshape(-1, 1)
        error = y[:len(yGuess)] - yGuess

        # Estimate a single sigma from residual variance
        self.sigma = np.sqrt((1 / X.shape[0]) * np.sum(error ** 2))

        # We find a single a and a single b to cover the space of our dimension
        a = np.min(y)
        b = np.max(y)

        # a will be min - range
        self.a = a - (b - a)

        # b is max + range
        self.b = b + (b - a)

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
        num_data = X.shape[0]
        num_batches = int(num_data / batch) + 1
        # For every batch of data, we get predictions
        preds = []
        for i in range(num_batches):
            batch_res = self.model(X[i * batch:min((i + 1) * batch, num_data)])
            # We concatenate them
            preds += list(batch_res.detach().cpu().numpy().ravel())

        preds = np.array(preds)
        preds = np.expand_dims(preds, axis=1)

        sigmas = np.stack([[self.sigma] * len(X)
                           for _ in range(1)],
                          axis=1)

        params_normal = np.concatenate((preds, sigmas), axis=1)

        # The first generative regressor is gaussian, the second is uniform
        # For the whole list of distributions, run
        #   import rampwf as rw
        #   rw.utils.distributions_dict
        types = np.array([[0, 1], ] * len(X))

        # Uniform
        a_array = np.array([self.a] * len(X))
        b_array = np.array([self.b] * len(X))
        params_uniform = np.stack((a_array, b_array), axis=1)

        # We give more weight to the gaussian one
        weights = np.array([[0.999, 0.001], ] * len(X))

        # We concatenate the params
        # To get information about the parameters of the distribution you are
        # using, you can run
        #   import rampwf as rw
        #   [(v,v.params) for v in rw.utils.distributions_dict.values()]
        params = np.concatenate((params_normal, params_uniform), axis=1)
        return weights, types, params


class PytorchReg(nn.Module):
    """
    A simple perceptron with two linear layers and one activation in between
    """
    def __init__(self, input_size):
        super(PytorchReg, self).__init__()
        layer_size = 15
        self.linear = nn.Linear(input_size, layer_size)
        self.act = nn.ReLU()
        self.linear_out = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear_out(x)
        return x


def train_model(model, dataset, optimizer, n_epochs=10, batch_size=128,
                loss_fn=nn.MSELoss()):
    """
    Training model over dataset with loss function loss_fn.
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
    """
    Tests the model over a dataset with loss function loss_fn and prints
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
        loss = loss_fn(y, out)
        print('Loss: {:.4f}'.format(loss.item()))
    return loss