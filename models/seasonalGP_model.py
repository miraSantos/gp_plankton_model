import gpytorch as gp
import gpytorch.kernels


class seasonalGPModel(gp.models.ExactGP):
    def __init__(self,
                 train_x,
                 train_y,
                 likelihood,
                 num_dims,
                 lt_l_prior,
                 lt_l_constraint,
                 lt_eps,
                 s_rbf_l_prior,
                 s_rbf_l_constraint,
                 s_rbf_eps,
                 s_pl_prior,
                 s_pl_constraint,
                 s_l_prior,
                 s_pk_l_constraint,
                 s_pk_eps,
                 wn_l_prior,
                 wn_l_constraint,
                 wn_a_constraint,
                 wn_eps
                 ):
        super(seasonalGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(
            gp.kernels.RBFKernel(
                ard_num_dims=num_dims,
                lengthscale_prior=lt_l_prior,
                lengthscale_constraint=lt_l_constraint,
                eps=lt_eps
                ) +
                (gp.kernels.RBFKernel(
                    ard_num_dims=num_dims,
                    lengthscale_prior= s_rbf_l_prior,
                    lengthscale_constraint= s_rbf_l_constraint,
                    eps= s_rbf_eps
                )* gp.kernels.PeriodicKernel(
                    ard_num_dims=num_dims,
                    period_length_prior= s_pl_prior,
                    period_length_constraint= s_pl_constraint,
                    lengthscale_prior= s_l_prior,
                    lengthscale_constraint= s_pk_l_constraint,
                    eps= s_pk_eps
                )) +
                gp.kernels.RQKernel(
                    ard_num_dims=num_dims,
                    lengthscale_prior=wn_l_prior,
                    lengthscale_constraint= wn_l_constraint,
                    alpha_constraint= wn_a_constraint ,
                    eps= wn_eps
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
