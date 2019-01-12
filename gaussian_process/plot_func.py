import matplotlib.pyplot as plt
import seaborn as sb


def get_plot(ax, f, x_domain, pred_x):

    se_process = gp.gaussian_process(
        mean=means.zero, kernel=kernels.squared_exponential(4, 2))

    ax.set_title("squared exponential process")

    y_domain = f(samples)

    se_process.set(x_domain, y_domain)

    pred_y = np.sin(pred_x)

    sem, sev = se_process.posterior(pred_x)

    sb.lineplot(pred_x, pred_y, ax=ax)
    sb.lineplot(pred_x, sem, ax=ax1)

    ax.fill_between(predict, sem - sev, sem + sev, alpha=0.2, color="k")

    return ax
