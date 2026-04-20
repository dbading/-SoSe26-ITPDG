import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

safe_gif = True

def parameters():
    return {
        "l": -2,
        "r": 2,
        # Example function. Replace directly if you want.
        # "f": lambda x: np.exp(-(x**2)) * (1.0 + 0.4 * np.cos(6.0 * x)),
        # "f": lambda x: np.where(np.abs(x + 0.5*(l + r)) <= 0.5*(r - l) <= 1, -np.abs(x)+1, 0.0),
        # "f": lambda x: np.where(np.abs(x + 0.5*(l + r)) <= 0.5*(r - l) <= 0.5, 1.0, 0.0),
        # "f": lambda x: np.where(np.abs(x + 0.5*(l + r)) <= 0.5*(r - l) , -np.abs(x)+1, 0.0),
        "f": lambda x: -np.abs(x) + r,
        # "f": lambda x: np.exp(-x**2),
        "n_plot": 1200,
        "n_integration": 5000,
        "n_frequencies": 320,
        "omega_max": 20.0,
        "interval_label": "[-2, 2]",
    }



def compute_fourier_transform_data(
    f,
    l,
    r,
    n_plot=1200,
    n_integration=5000,
    n_frequencies=320,
    omega_max=20.0,
):
    r"""
    Precompute everything once and cache it for the animation.

    We use the Fourier transform convention
        F(omega) = \int_l^r f(x) e^{-i omega x} dx
                 = \int_l^r f(x) cos(omega x) dx
                   - i \int_l^r f(x) sin(omega x) dx.

    As in the Fourier-series and convolution code, we separate the grids:
    - x_plot is used for visualization
    - x_int is used for the numerical integral

    Since we integrate only on [l, r], this is the Fourier transform of the
    truncated function on that interval.
    """
    x_plot = np.linspace(l, r, n_plot)
    x_int = np.linspace(l, r, n_integration)

    f_plot = f(x_plot)
    f_int = f(x_int)

    omega_values = np.linspace(-omega_max, omega_max, n_frequencies)

    kernel_real_list = []
    kernel_imag_list = []
    integrand_real_list = []
    integrand_imag_list = []

    transform_real = np.zeros_like(omega_values)
    transform_imag = np.zeros_like(omega_values)

    for i, omega in enumerate(omega_values):
        kernel_real_plot = np.cos(omega * x_plot)
        kernel_imag_plot = -np.sin(omega * x_plot)

        integrand_real_plot = f_plot * kernel_real_plot
        integrand_imag_plot = f_plot * kernel_imag_plot

        kernel_real_list.append(kernel_real_plot)
        kernel_imag_list.append(kernel_imag_plot)
        integrand_real_list.append(integrand_real_plot)
        integrand_imag_list.append(integrand_imag_plot)

        transform_real[i] = np.trapezoid(f_int * np.cos(omega * x_int), x_int)
        transform_imag[i] = np.trapezoid(f_int * (-np.sin(omega * x_int)), x_int)

    transform_abs = np.sqrt(transform_real**2 + transform_imag**2)

    return {
        "x_plot": x_plot,
        "f_plot": f_plot,
        "omega_values": omega_values,
        "kernel_real_list": kernel_real_list,
        "kernel_imag_list": kernel_imag_list,
        "integrand_real_list": integrand_real_list,
        "integrand_imag_list": integrand_imag_list,
        "transform_real": transform_real,
        "transform_imag": transform_imag,
        "transform_abs": transform_abs,
    }



def plot_static_overview(data):
    x_plot = data["x_plot"]
    f_plot = data["f_plot"]
    omega_values = data["omega_values"]
    transform_real = data["transform_real"]
    transform_imag = data["transform_imag"]
    transform_abs = data["transform_abs"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(x_plot, f_plot, linewidth=2, label="f(x)")
    axes[0].set_title("Original function")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(omega_values, transform_real, linewidth=2, label=r"$\Re(\hat f(\omega))$")
    axes[1].plot(omega_values, transform_imag, linewidth=2, label=r"$\Im(\hat f(\omega))$")
    axes[1].plot(omega_values, transform_abs, linewidth=2, label=r"$|\hat f(\omega)|$")
    axes[1].set_title("Fourier transform")
    axes[1].set_xlabel(r"$\omega$")
    axes[1].set_ylabel(r"$\hat f(\omega)$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def animate_fourier_transform(data, interval=50, repeat=True, save_path=None, fps=20):
    x_plot = data["x_plot"]
    f_plot = data["f_plot"]
    omega_values = data["omega_values"]
    kernel_real_list = data["kernel_real_list"]
    kernel_imag_list = data["kernel_imag_list"]
    integrand_real_list = data["integrand_real_list"]
    integrand_imag_list = data["integrand_imag_list"]
    transform_real = data["transform_real"]
    transform_imag = data["transform_imag"]
    transform_abs = data["transform_abs"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax1, ax2, ax3 = axes

    # Panel 1: fixed f and oscillatory kernel parts
    line_f, = ax1.plot(x_plot, f_plot, linewidth=2, label="f(x)")
    line_kernel_real, = ax1.plot(x_plot, kernel_real_list[0], linewidth=2, label=r"$\cos(\omega x)$")
    line_kernel_imag, = ax1.plot(x_plot, kernel_imag_list[0], linewidth=2, label=r"$-\sin(\omega x)$")
    ax1.set_title(r"Function and Fourier kernel for $\omega = {:.2f}$".format(omega_values[0]))
    ax1.set_xlabel("x")
    ax1.set_ylabel("value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: current real and imaginary integrands
    line_integrand_real, = ax2.plot(x_plot, integrand_real_list[0], linewidth=2, label=r"$f(x)\cos(\omega x)$")
    line_integrand_imag, = ax2.plot(x_plot, integrand_imag_list[0], linewidth=2, label=r"$-f(x)\sin(\omega x)$")
    ax2.set_title(r"Current integrands for $\omega = {:.2f}$".format(omega_values[0]))
    ax2.set_xlabel("x")
    ax2.set_ylabel("value")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: transform built up while omega is swept
    line_transform_real, = ax3.plot([], [], linewidth=2, label=r"$\Re(\hat f(\omega))$")
    line_transform_imag, = ax3.plot([], [], linewidth=2, label=r"$\Im(\hat f(\omega))$")
    line_transform_abs, = ax3.plot([], [], linewidth=2, label=r"$|\hat f(\omega)|$")
    current_real_point, = ax3.plot([omega_values[0]], [transform_real[0]], marker="o", linestyle="None")
    current_imag_point, = ax3.plot([omega_values[0]], [transform_imag[0]], marker="o", linestyle="None")
    current_abs_point, = ax3.plot([omega_values[0]], [transform_abs[0]], marker="o", linestyle="None")
    ax3.set_title("Fourier transform built up frequency by frequency")
    ax3.set_xlabel(r"$\omega$")
    ax3.set_ylabel(r"$\hat f(\omega)$")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Stable axis limits
    y1_min = min(np.min(f_plot), min(np.min(arr) for arr in kernel_real_list), min(np.min(arr) for arr in kernel_imag_list))
    y1_max = max(np.max(f_plot), max(np.max(arr) for arr in kernel_real_list), max(np.max(arr) for arr in kernel_imag_list))
    m1 = 0.1 * (y1_max - y1_min + 1e-12)
    ax1.set_xlim(x_plot[0] - 1, x_plot[-1] + 1)
    ax1.set_ylim(y1_min - m1, y1_max + m1)

    y2_min = min(min(np.min(arr) for arr in integrand_real_list), min(np.min(arr) for arr in integrand_imag_list))
    y2_max = max(max(np.max(arr) for arr in integrand_real_list), max(np.max(arr) for arr in integrand_imag_list))
    m2 = 0.1 * (y2_max - y2_min + 1e-12)
    ax2.set_xlim(x_plot[0] - 1, x_plot[-1] + 1)
    ax2.set_ylim(y2_min - m2, y2_max + m2)

    y3_min = min(np.min(transform_real), np.min(transform_imag), np.min(transform_abs))
    y3_max = max(np.max(transform_real), np.max(transform_imag), np.max(transform_abs))
    m3 = 0.1 * (y3_max - y3_min + 1e-12)
    ax3.set_xlim(omega_values[0] - 1, omega_values[-1] + 1)
    ax3.set_ylim(y3_min - m3, y3_max + m3)

    def update(frame):
        omega = omega_values[frame]

        line_kernel_real.set_ydata(kernel_real_list[frame])
        line_kernel_imag.set_ydata(kernel_imag_list[frame])
        line_integrand_real.set_ydata(integrand_real_list[frame])
        line_integrand_imag.set_ydata(integrand_imag_list[frame])

        line_transform_real.set_data(omega_values[: frame + 1], transform_real[: frame + 1])
        line_transform_imag.set_data(omega_values[: frame + 1], transform_imag[: frame + 1])
        line_transform_abs.set_data(omega_values[: frame + 1], transform_abs[: frame + 1])

        current_real_point.set_data([omega], [transform_real[frame]])
        current_imag_point.set_data([omega], [transform_imag[frame]])
        current_abs_point.set_data([omega], [transform_abs[frame]])

        ax1.set_title(rf"Function and Fourier kernel for $\omega = {omega:.2f}$")
        ax2.set_title(rf"Current integrands for $\omega = {omega:.2f}$")
        ax3.set_title(
            rf"$\Re(\hat f(\omega)) = {transform_real[frame]:.4f}$, "
            rf"$\Im(\hat f(\omega)) = {transform_imag[frame]:.4f}$, "
            rf"$|\hat f(\omega)| = {transform_abs[frame]:.4f}$"
        )

        return (
            line_kernel_real,
            line_kernel_imag,
            line_integrand_real,
            line_integrand_imag,
            line_transform_real,
            line_transform_imag,
            line_transform_abs,
            current_real_point,
            current_imag_point,
            current_abs_point,
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(omega_values),
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            ani.save(save_path, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    plt.show()
    return ani



if __name__ == "__main__":
    params = parameters()
    l = params["l"]
    r = params["r"]
    g = params["f"]
    # auskommentiere 
    f = lambda x: np.where(np.abs(x + 0.5*(l + r)) <= 0.5*(r - l) , g(x), 0.0)
    # 
    n_plot = params["n_plot"]
    n_integration = params["n_integration"]
    n_frequencies = params["n_frequencies"]
    omega_max = params["omega_max"]

    data = compute_fourier_transform_data(
        f=f,
        l=l,
        r=r,
        n_plot=n_plot,
        n_integration=n_integration,
        n_frequencies=n_frequencies,
        omega_max=omega_max,
    )

    # Static overview
    plot_static_overview(data)

    # Animation
    ani = animate_fourier_transform(data, interval=40)

    # Optional: save animation
    if safe_gif == True:
        from pathlib import Path

        gifs_dir = Path("./gifs")
        gifs_dir.mkdir(exist_ok=True)

        i = 1
        while (gifs_dir / f"fourier_transform_animation_{i}.gif").exists():
            i += 1

        save_path = gifs_dir / f"fourier_transform_animation_{i}.gif"
        print(save_path)
        ani = animate_fourier_transform(data, interval=40, save_path=str(save_path), fps=20)
        
        # ani = animate_fourier_transform(data, interval=40, save_path="fourier_transform_animation.mp4", fps=20)
