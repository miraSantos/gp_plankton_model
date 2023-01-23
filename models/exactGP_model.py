import gpytorch
import gpytorch.kernels
import gpytorch.constraints

# simplest form of GP model, exact inference
class exactGP_model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(exactGP_model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
