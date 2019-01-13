import numpy as np
import means
import kernels
import gaussian_process as gp
import matplotlib.pyplot as plt
import seaborn as sb

def f(X):
    return X * np.sin(X / 2) + X 

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

linear_process = gp.gaussian_process(mean=means.zero, kernel=kernels.linear)

se_process = gp.gaussian_process(
    mean=means.zero, kernel=kernels.squared_exponential(4, 2))

bm_process = gp.gaussian_process(
    mean=means.zero, kernel=kernels.brownian_motion(30))

lp_process = gp.gaussian_process(mean=means.zero, kernel=kernels.Lp(5))

ax1.set_title("linear process")
ax2.set_title("squared exponential process")
ax3.set_title("Brownian Motion Ish")
ax4.set_title("Lp Kernel")

x_domain = np.arange(-5, 6, 1)
y_domain = f(x_domain)

linear_process.set(x_domain, y_domain)
se_process.set(x_domain, y_domain)
bm_process.set(x_domain, y_domain)
lp_process.set(x_domain, y_domain)

predict = np.arange(-5, 20, 0.1)
pred_y = f(predict)

lpm, lpv = linear_process.posterior(predict)
sem, sev = se_process.posterior(predict)
bmm, bmv = bm_process.posterior(predict)
lm, lv = lp_process.posterior(predict)

sb.lineplot(predict, pred_y, ax=ax1)
sb.lineplot(predict, pred_y, ax=ax2)
sb.lineplot(predict, pred_y, ax=ax3)
sb.lineplot(predict, pred_y, ax=ax4)

sb.lineplot(predict, lpm, ax=ax1)
sb.lineplot(predict, sem, ax=ax2)
sb.lineplot(predict, bmm, ax=ax3)
sb.lineplot(predict, lm, ax=ax4)

ax1.fill_between(predict, lpm - lpv, lpm + lpv, alpha=0.2, color="k")
ax2.fill_between(predict, sem - sev, sem + sev, alpha=0.2, color="k")
ax3.fill_between(predict, bmm - bmv, bmm + bmv, alpha=0.2, color="k")
ax4.fill_between(predict, lm - lv, lm + lv, alpha=0.2, color="k")

plt.tight_layout()
plt.show()
