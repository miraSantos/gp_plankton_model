import gpytorch
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(x_test, likelihood, model):
    """
    :param x_test:
    :param likelihood:
    :param model:
    :return:
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(X_train))

        # Initialize plot
        width = 20
        height = 5
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training preprocess as black stars
        ax.plot(df.date[:len(X_train)], y_train, 'k*',label="training preprocess")
        # Plot predictive means as blue line
        ax.plot(df.date[:len(X_train)], observed_pred.mean, 'b', label="prediction")
        ax.plot(df.date[len(X_train):], y_test, 'g', label="testing preprocess")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(df.date[len(X_train):], lower, upper, alpha=0.5)
        ax.xaxis.set_major_locator(YearLocator(base=1))
        ax.legend()
        ax.grid()
        eval_img = res_path+'evaluation.png'
        ax.set_title("Evaluation")
        fig.savefig(eval_img)
        im = Image.open(eval_img)
        wandb.log({"Evaluation": wandb.Image(im)})


if __name__ == '__main__':
