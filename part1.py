import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class MyLinearRegression:
    def __init__(self, weights_init='random', add_bias=True, learning_rate=1e-5,
                 num_iterations=1_000, verbose=False, max_error=1e-5):
        """ Linear regression model using gradient descent

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int
                maximum number of iterations in gradient descent
            verbose: bool
                enabling verbose output
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
        """

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error

        self.weights = None
        self.bias = None

    def initialize_weights(self, n_features):
        """ weights initialization function """
        if self.weights_init == 'random':
            return np.random.uniform(low=0.0001, high=1.0, size=(n_features, 1))
        elif self.weights_init == 'zeros':
            return np.zeros((n_features, 1))
        else:
            raise NotImplementedError

    def cost(self, target, pred):
        """ calculate cost function

            # Arguments:
                target: np.array
                    array of target floating point numbers
                pred: np.array
                    array of predicted floating points numbers
        """

        loss = np.mean((target - pred) ** 2)
        if self.verbose:
            print('loss: {}'.format(loss)
                  + 'for target: {}'.format(target)
                  + 'and pred: {}'.format(pred))
        return loss

    def fit(self, x, y):
        self.bias = 0
        self.weights = self.initialize_weights(x.shape[1])

        for i in range(self.num_iterations):
            # step 1: calculate current_loss value
            y_pred = self.predict(x)
            current_loss = self.cost(y, y_pred)

            # step 2: calculate gradient value
            dw = (1 / x.shape[0]) * np.dot(x.T, (y_pred - y))
            db = (1 / x.shape[0]) * np.sum(y_pred - y)

            # step 3: update weights using learning rate and gradient value
            self.weights -= self.learning_rate * dw
            if self.add_bias:
                self.bias -= self.learning_rate * db

            # step 4: calculate new_loss value
            new_y_pred = np.dot(x, self.weights) + self.bias
            new_loss = self.cost(y, new_y_pred)

            if self.verbose and i % 100 == 0:
                print(f"Iteration: {i}, loss: {new_loss}")

            # step 5: if new_loss and current_loss difference is greater than max_error -> break;
            #         if iteration is greater than max_iterations -> break
            diff_loss = np.abs(new_loss - current_loss)
            if diff_loss < self.max_error:
                print(f"Convergence reached after {i} iterations.")
                break

    def predict(self, x):
        """ prediction function """
        return x.dot(self.weights) + self.bias


def normal_equation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))


if __name__ == "__main__":
    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100, 1)

    # normalization of input data
    x /= np.max(x)

    plt.title('Data samples')
    plt.scatter(x, y)
    plt.savefig('data_samples.png')

    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title('Data samples with sklearn model')
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color='r')
    plt.savefig('part1/sklearn_model.png')
    print('Sklearn MSE: ', mean_squared_error(y, y_hat_sklearn))

    # Your linear regression model
    my_model = MyLinearRegression(num_iterations=1_000_000, verbose=False)
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title('Data samples with my model')
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r')
    plt.savefig('part1/my_model.png')
    print('My MSE: ', mean_squared_error(y, y_hat))

    # Normal equation
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title('Data samples with normal equation')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color='r')
    plt.savefig('part1/normal_equation.png')
    print('Normal equation MSE: ', mean_squared_error(y, y_hat_normal))
