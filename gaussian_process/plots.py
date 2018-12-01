import numpy as np
import means
import kernels
import gaussian_process as gp
import matplotlib.pyplot as plt
import seaborn as sb

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

linear_process = gp.gaussian_process(mean=means.zero, kernel=kernels.linear)

se_process = gp.gaussian_process(
    mean=means.zero, kernel=kernels.squared_exponential(2, 100))

bm_process = gp.gaussian_process(
    mean=means.zero, kernel=kernels.brownian_motion(1))

periodic_process = gp.gaussian_process(
    mean=means.zero, kernel=kernels.periodic(1, 5))


ax1.set_title("linear process")
ax2.set_title("squared exponential process")
ax3.set_title("brownian motion process")
ax4.set_title("periodic process")


x_domain = np.arange(0, 1, 0.005)

linear_process.set(x_domain)
se_process.set(x_domain)
bm_process.set(x_domain)
periodic_process.set(x_domain)

for _ in range(5):
    sb.lineplot(x=x_domain, y=linear_process.sample(), ax=ax1)
    sb.lineplot(x=x_domain, y=se_process.sample(), ax=ax2)
    sb.lineplot(x=x_domain, y=bm_process.sample(), ax=ax3)
    sb.lineplot(x=x_domain, y=periodic_process.sample(), ax=ax4)


plt.show()
