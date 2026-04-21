import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

safe_gif = True

#####################################
# define functions to convolve here #
#####################################

# def box(x):
#     T = 0.5
#     return (x > -T) * (x < T) * 1.0

# def step(x):
#     return (x > 0) * 1.0

# def biphasic_unitary_step(x):
#     return box(x - 0.5) - box(x - 1.5)

# def exponential(x):
#     return np.exp(x)

# def triangle(x):
#     y = ((x + 1) * step(+1) - 2 * x * step(x)) * box(1 / 2 * x)
#     return y

# def exponential_decay(x):
#     return np.exp(-x)

# def positive_exponential_decay(x):
#     return np.exp(-x) * step(x)

# def sinusoid_1_period(x):
#     period = np.pi
#     box_stretch = np.pi
#     box_shift = box_stretch / 2
#     y = np.sin(x * (2 * np.pi) / period) * box(1 / box_stretch * (x - box_shift))
#     return y

# def sinusoid_3_period_phase(x):
#     period = 1 / 3 * np.pi
#     phase_shift = 0
#     box_stretch = np.pi
#     box_shift = box_stretch / 2
#     y = np.sin(x * (2 * np.pi) / period + phase_shift) * box(1 / box_stretch * (x - box_shift))
#     return y

# def sinusoid_3_period_phase_shifted(x):
#     period = 1 / 3 * np.pi
#     phase_shift = np.pi
#     box_stretch = np.pi
#     box_shift = box_stretch / 2
#     y = np.sin(x * (2 * np.pi) / period + phase_shift) * box(1 / box_stretch * (x - box_shift))
#     return y

# def M_1(x):
#     return np.where(np.abs(x) < 1, 1 - np.abs(x), 0)

# def M_k(k, x):
#     if k == 1:
#         return M_1(x)
#     else:
#         return np.convolve(M_1(x), M_k(k - 1, x), mode='same') * (x[1] - x[0])

# def tut_3_4(x):
#     return 1 / (1 + x**2)

# def tut_3_4_b(x):
#     return np.exp(-np.abs(x))

# def your_function(x):
#     y_value = 1.0 * M_k(4, x)
#     return y_value


def parameters():
    return {
        "f_l": 0,
        "f_r": 1,
        # function on the interval [f_l, f_r]
        "f": lambda x: 1,
        "g_l": 0,
        "g_r": 1,
        "g": lambda x: 1,
        "x_min": -.5,
        "x_max": 2.5,
        "n_plot": 1200,
        "n_integration": 5000,
        "interval_label": "[-2, 2]",
    }


def evaluate_supported_function(func, x, left, right):
    """
    Evaluate func on x and restrict it to the support interval [left, right].
    Scalars returned by func are automatically broadcast to the shape of x.
    """
    x = np.asarray(x)
    values = func(x)
    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        values = np.full_like(x, float(values), dtype=float)

    mask = (x >= left) & (x <= right)
    return np.where(mask, values, 0.0)


def compute_convolution_values(f, g, shifts, f_l, f_r, g_l, g_r, n_integration):
    """
    Compute (f * g)(x) = ∫ f(t) g(x - t) dt numerically.
    The integral is taken over the support interval of f, because outside this
    interval the integrand vanishes by construction.
    """
    t = np.linspace(f_l, f_r, n_integration, endpoint=False)
    dt = t[1] - t[0]
    ft = evaluate_supported_function(f, t, f_l, f_r)

    convolution_values = np.zeros_like(shifts, dtype=float)
    for i, shift in enumerate(shifts):
        g_shifted = evaluate_supported_function(g, shift - t, g_l, g_r)
        convolution_values[i] = np.sum(ft * g_shifted) * dt

    return convolution_values


def next_gif_path(base_dir, prefix="faltung_animation"):
    from pathlib import Path

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while (base_dir / f"{prefix}_{i}.gif").exists():
        i += 1

    return base_dir / f"{prefix}_{i}.gif"


def animate_convolution(
    f,
    g,
    shifts,
    f_l,
    f_r,
    g_l,
    g_r,
    n_integration,
    interval=40,
    repeat=True,
    save_path=None,
    fps=20,
):
    """
    Animate the convolution by showing
    - top: f(t) and the shifted/reflected function g(x - t)
    - bottom: the convolution curve and the current point
    """
    t_plot_min = min(f_l, shifts.min() - g_r) - 0.25
    t_plot_max = max(f_r, shifts.max() - g_l) + 0.25
    t_plot = np.linspace(t_plot_min, t_plot_max, 1500)

    f_plot = evaluate_supported_function(f, t_plot, f_l, f_r)
    convolution_values = compute_convolution_values(
        f, g, shifts, f_l, f_r, g_l, g_r, n_integration
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top plot: f and shifted g
    f_line, = ax_top.plot(t_plot, f_plot, linewidth=2, label="f(t)")
    g_shifted_initial = evaluate_supported_function(g, shifts[0] - t_plot, g_l, g_r)
    g_line, = ax_top.plot(t_plot, g_shifted_initial, linewidth=2, label=r"g(x-t)")

    product_initial = f_plot * g_shifted_initial
    product_fill = ax_top.fill_between(
        t_plot, 0, product_initial, alpha=0.25, label=r"f(t)g(x-t)"
    )

    y_top_min = min(0.0, np.min(f_plot), np.min(g_shifted_initial))
    y_top_max = max(0.0, np.max(f_plot), np.max(g_shifted_initial))
    y_top_margin = 0.15 * (y_top_max - y_top_min + 1e-12)

    ax_top.set_xlim(t_plot[0], t_plot[-1])
    ax_top.set_ylim(y_top_min - y_top_margin, y_top_max + y_top_margin)
    ax_top.set_xlabel("t")
    ax_top.set_ylabel("value")
    ax_top.set_title("Convolution animation")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    # Bottom plot: convolution curve
    conv_line, = ax_bottom.plot(shifts, convolution_values, linewidth=2, label=r"(f * g)(x)")
    current_point, = ax_bottom.plot([shifts[0]], [convolution_values[0]], marker="o", linestyle="None")
    traced_line, = ax_bottom.plot([shifts[0]], [convolution_values[0]], linewidth=2, alpha=0.8)

    y_bottom_min = min(0.0, np.min(convolution_values))
    y_bottom_max = max(0.0, np.max(convolution_values))
    y_bottom_margin = 0.15 * (y_bottom_max - y_bottom_min + 1e-12)

    ax_bottom.set_xlim(shifts[0], shifts[-1])
    ax_bottom.set_ylim(y_bottom_min - y_bottom_margin, y_bottom_max + y_bottom_margin)
    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel(r"(f * g)(x)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend()

    def update(frame):
        nonlocal product_fill

        shift = shifts[frame]
        g_shifted = evaluate_supported_function(g, shift - t_plot, g_l, g_r)
        product = f_plot * g_shifted

        g_line.set_ydata(g_shifted)

        product_fill.remove()
        product_fill = ax_top.fill_between(t_plot, 0, product, alpha=0.25)

        current_point.set_data([shift], [convolution_values[frame]])
        traced_line.set_data(shifts[: frame + 1], convolution_values[: frame + 1])
        ax_top.set_title(f"Convolution animation at shift x = {shift:.3f}")

        return g_line, current_point, traced_line, product_fill

    ani = FuncAnimation(
        fig,
        update,
        frames=len(shifts),
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    plt.tight_layout()

    if save_path is not None:
        ani.save(save_path, writer="pillow", fps=fps)
        print(f"GIF saved to: {save_path}")

    plt.show()
    return ani


if __name__ == "__main__":
    params = parameters()

    shifts = np.linspace(params["x_min"], params["x_max"], params["n_plot"])

    if safe_gif:
        save_path = next_gif_path("./Faltung/gifs", prefix="faltung_animation")
        ani = animate_convolution(
            params["f"],
            params["g"],
            shifts,
            params["f_l"],
            params["f_r"],
            params["g_l"],
            params["g_r"],
            params["n_integration"],
            interval=40,
            repeat=True,
            save_path=str(save_path),
            fps=20,
        )
    else:
        ani = animate_convolution(
            params["f"],
            params["g"],
            shifts,
            params["f_l"],
            params["f_r"],
            params["g_l"],
            params["g_r"],
            params["n_integration"],
            interval=40,
            repeat=True,
        )
