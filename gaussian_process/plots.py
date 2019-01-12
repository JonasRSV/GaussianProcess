import numpy as np
import means
import kernels
import gaussian_process as gp
import matplotlib.pyplot as plt
import seaborn as sb

fig, (ax1, ax2) = plt.subplots(2, 1)

linear_process = gp.gaussian_process(mean=means.zero, kernel=kernels.linear)

se_process = gp.gaussian_process(
    mean=means.zero, kernel=kernels.squared_exponential(4, 2))

ax1.set_title("linear process")
ax2.set_title("squared exponential process")

x_domain = np.arange(-5, 5, 2)
y_domain = np.sin(x_domain)

linear_process.set(x_domain, y_domain)
se_process.set(x_domain, y_domain)

predict = np.arange(-5, 5, 0.1)
pred_y = np.sin(predict)

lpm, lpv = linear_process.posterior(predict)
sem, sev = se_process.posterior(predict)

sb.lineplot(predict, pred_y, ax=ax1)
sb.lineplot(predict, pred_y, ax=ax2)

sb.lineplot(predict, lpm, ax=ax1)
sb.lineplot(predict, sem, ax=ax2)

ax1.fill_between(predict, lpm - lpv, lpm + lpv, alpha=0.2, color="k")
ax2.fill_between(predict, sem - sev, sem + sev, alpha=0.2, color="k")

plt.tight_layout()
plt.show()
