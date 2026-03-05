import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gamma


def main():
    ds = xr.open_dataset("data/GPCP/precip.mon.mean.error.nc")

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    std_est = ds.mean(["time", "lon"]).precip
    std_est.plot(ax=axes[0])
    plt.title("GPCP Precipitation Observation Error Estimate")
    plt.ylabel("Absolute Error Estimate (mm/day)")

    var_est = (ds.precip**2).mean(["time", "lon"])
    var_est.plot(ax=axes[1])
    plt.title("GPCP Precipitation Observation Error Estimate Squared")
    plt.ylabel("Variance Estimate (mm$^2$/day$^2$)")

    var_est.plot.hist(ax=axes[2], density=True)
    plt.title("Histogram of Variance Estimate")
    plt.xlabel("Variance Estimate (mm$^2$/day$^2$)")
    plt.ylabel("Density")
    x = np.linspace(0, 2, 100)
    y = gamma.pdf(x, a=3, scale=0.3)
    # y = gamma.pdf(x, a=4, scale=0.225)
    axes[2].plot(x, y, "r--", label="Gamma PDF")

    print("Mean variance estimate:", var_est.mean().item())

    plt.show()


if __name__ == "__main__":
    main()
