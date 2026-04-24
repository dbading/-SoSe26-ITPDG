"""
Faltung / convolution animation with faster GIF export.

Main changes compared to the previous version:
- GIF saving is enabled by default via SAVE_GIF = True.
- The number of GIF frames is separated from the number of curve points.
- The filled product area is updated instead of recreated in every frame.
- The convolution integral is computed in vectorized chunks.
- The GIF is saved into a local ./gifs folder next to this Python file.

Required packages:
    pip install numpy matplotlib pillow
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon


# -----------------------------------------------------------------------------
# Export / display switches
# -----------------------------------------------------------------------------

SAVE_GIF = False
SHOW_ANIMATION_AFTER_SAVING = True


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
#         return np.convolve(M_1(x), M_k(k - 1, x), mode="same") * (x[1] - x[0])

# def tut_3_4(x):
#     return 1 / (1 + x**2)

# def tut_3_4_b(x):
#     return np.exp(-np.abs(x))

# def your_function(x):
#     y_value = 1.0 * M_k(4, x)
#     return y_value


def parameters():
    return {
        # Support interval of f
        "f_l": 0,
        "f_r": 1,
        "f": lambda x: 1,

        # Support interval of g
        "g_l": 0,
        "g_r": 1,
        "g": lambda x: 1,

        # x-range for the convolution curve
        "x_min": -0.5,
        "x_max": 2.5,

        # Numerical resolution of the displayed convolution curve.
        # This controls the accuracy/smoothness of the lower plot.
        "n_curve": 600,

        # Numerical resolution of the integral in t.
        # Larger = more accurate but slower. 1200 is usually enough for animations.
        "n_integration": 1200,

        # Number of actual GIF frames.
        # This is the most important speed parameter for saving.
        "n_frames": 180,

        # Number of points used for drawing f(t), g(x-t), and the product.
        "n_t_plot": 800,

        # Animation / GIF settings
        "interval": 40,      # ms between frames for interactive display
        "fps": 20,           # frames per second in the exported GIF
        "dpi": 100,          # lower dpi = smaller and faster GIF
        "repeat": True,

        # File output
        "output_folder": "gifs",
        "output_prefix": "faltung_animation",
    }


def evaluate_supported_function(func, x, left, right):
    """
    Evaluate func on x and restrict it to the support interval [left, right].

    Important detail:
    If func returns a scalar, for example lambda x: 1, the scalar is broadcast
    to the shape of x. This avoids shape errors for constant functions.
    """
    x = np.asarray(x)
    values = func(x)
    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        values = np.full_like(x, float(values), dtype=float)

    mask = (x >= left) & (x <= right)
    return np.where(mask, values, 0.0)


def compute_convolution_values(
    f,
    g,
    shifts,
    f_l,
    f_r,
    g_l,
    g_r,
    n_integration,
    chunk_size=128,
):
    r"""
    Compute convolution values numerically:

        (f * g)(x) = \int f(t) g(x - t) dt.

    The integral is evaluated over the support interval of f. For speed, the
    shifts are processed in vectorized chunks instead of one Python loop per x.
    """
    shifts = np.asarray(shifts, dtype=float)

    t = np.linspace(f_l, f_r, n_integration)
    dt = t[1] - t[0]
    ft = evaluate_supported_function(f, t, f_l, f_r)

    convolution_values = np.zeros_like(shifts, dtype=float)

    for start in range(0, len(shifts), chunk_size):
        end = min(start + chunk_size, len(shifts))
        shift_chunk = shifts[start:end, None]

        # Shape: (number_of_shifts_in_chunk, n_integration)
        g_shifted = evaluate_supported_function(g, shift_chunk - t[None, :], g_l, g_r)

        # Trapezoidal rule along t-axis.
        integrand = ft[None, :] * g_shifted
        convolution_values[start:end] = np.trapz(integrand, dx=dt, axis=1)

    return convolution_values


def next_gif_path(base_dir, prefix="faltung_animation"):
    """
    Return a non-existing path like faltung_animation_1.gif,
    faltung_animation_2.gif, ... without overwriting older files.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while (base_dir / f"{prefix}_{i}.gif").exists():
        i += 1

    return base_dir / f"{prefix}_{i}.gif"


def fill_between_vertices(x, y):
    """
    Vertices for a polygon that fills the area between y=0 and y=y(x).
    This lets us update one existing Polygon instead of recreating fill_between
    in every animation frame.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.column_stack([
        np.r_[x, x[::-1]],
        np.r_[np.zeros_like(x), y[::-1]],
    ])


def animate_convolution(
    f,
    g,
    curve_shifts,
    f_l,
    f_r,
    g_l,
    g_r,
    n_integration,
    n_frames=180,
    n_t_plot=800,
    interval=40,
    repeat=True,
    save_path=None,
    fps=20,
    dpi=100,
    show=True,
):
    """
    Animate the convolution by showing
    - top: f(t), shifted/reflected g(x-t), and product f(t)g(x-t),
    - bottom: the convolution curve and the currently traced point.
    """
    curve_shifts = np.asarray(curve_shifts, dtype=float)

    print("Computing convolution curve ...")
    convolution_values = compute_convolution_values(
        f, g, curve_shifts, f_l, f_r, g_l, g_r, n_integration
    )

    # Use only a subset of the curve points as actual animation frames.
    n_frames = int(min(max(2, n_frames), len(curve_shifts)))
    frame_indices = np.linspace(0, len(curve_shifts) - 1, n_frames, dtype=int)

    t_plot_min = min(f_l, curve_shifts.min() - g_r) - 0.25
    t_plot_max = max(f_r, curve_shifts.max() - g_l) + 0.25
    t_plot = np.linspace(t_plot_min, t_plot_max, n_t_plot)

    f_plot = evaluate_supported_function(f, t_plot, f_l, f_r)

    # Estimate stable y-limits for the upper plot from f and g on their supports.
    g_support_grid = np.linspace(g_l, g_r, n_t_plot)
    g_values_on_support = evaluate_supported_function(g, g_support_grid, g_l, g_r)
    y_top_min = min(0.0, float(np.min(f_plot)), float(np.min(g_values_on_support)))
    y_top_max = max(0.0, float(np.max(f_plot)), float(np.max(g_values_on_support)))
    y_top_margin = 0.15 * (y_top_max - y_top_min + 1e-12)

    first_shift = curve_shifts[frame_indices[0]]
    g_shifted_initial = evaluate_supported_function(g, first_shift - t_plot, g_l, g_r)
    product_initial = f_plot * g_shifted_initial

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Top plot: f and shifted g
    ax_top.plot(t_plot, f_plot, linewidth=2, label="f(t)")
    g_line, = ax_top.plot(t_plot, g_shifted_initial, linewidth=2, label=r"g(x-t)")

    product_patch = Polygon(
        fill_between_vertices(t_plot, product_initial),
        closed=True,
        alpha=0.25,
        label=r"f(t)g(x-t)",
    )
    ax_top.add_patch(product_patch)

    ax_top.set_xlim(t_plot[0], t_plot[-1])
    ax_top.set_ylim(y_top_min - y_top_margin, y_top_max + y_top_margin)
    ax_top.set_xlabel("t")
    ax_top.set_ylabel("value")
    ax_top.set_title(f"Convolution animation at shift x = {first_shift:.3f}")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    # Bottom plot: convolution curve
    ax_bottom.plot(curve_shifts, convolution_values, linewidth=2, label=r"(f * g)(x)")
    current_point, = ax_bottom.plot(
        [first_shift],
        [convolution_values[frame_indices[0]]],
        marker="o",
        linestyle="None",
    )
    traced_line, = ax_bottom.plot(
        [first_shift],
        [convolution_values[frame_indices[0]]],
        linewidth=2,
        alpha=0.8,
    )

    y_bottom_min = min(0.0, float(np.min(convolution_values)))
    y_bottom_max = max(0.0, float(np.max(convolution_values)))
    y_bottom_margin = 0.15 * (y_bottom_max - y_bottom_min + 1e-12)

    ax_bottom.set_xlim(curve_shifts[0], curve_shifts[-1])
    ax_bottom.set_ylim(y_bottom_min - y_bottom_margin, y_bottom_max + y_bottom_margin)
    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel(r"(f * g)(x)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend()

    def update(frame_number):
        curve_index = frame_indices[frame_number]
        shift = curve_shifts[curve_index]

        g_shifted = evaluate_supported_function(g, shift - t_plot, g_l, g_r)
        product = f_plot * g_shifted

        g_line.set_ydata(g_shifted)
        product_patch.set_xy(fill_between_vertices(t_plot, product))

        current_point.set_data([shift], [convolution_values[curve_index]])
        traced_indices = frame_indices[: frame_number + 1]
        traced_line.set_data(curve_shifts[traced_indices], convolution_values[traced_indices])
        ax_top.set_title(f"Convolution animation at shift x = {shift:.3f}")

        return g_line, product_patch, current_point, traced_line

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        blit=False,
        repeat=repeat,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving GIF with {len(frame_indices)} frames to: {save_path}")

        def progress_callback(current_frame, total_frames):
            frame_number = current_frame + 1
            if frame_number == 1 or frame_number % 25 == 0 or frame_number == total_frames:
                print(f"  saved frame {frame_number}/{total_frames}")

        ani.save(
            str(save_path),
            writer=PillowWriter(fps=fps),
            dpi=dpi,
            progress_callback=progress_callback,
        )
        print(f"GIF saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani


if __name__ == "__main__":
    params = parameters()

    curve_shifts = np.linspace(
        params["x_min"],
        params["x_max"],
        params["n_curve"],
    )

    script_dir = Path(__file__).resolve().parent

    if SAVE_GIF:
        save_path = next_gif_path(
            script_dir / params["output_folder"],
            prefix=params["output_prefix"],
        )
    else:
        save_path = None

    animate_convolution(
        params["f"],
        params["g"],
        curve_shifts,
        params["f_l"],
        params["f_r"],
        params["g_l"],
        params["g_r"],
        params["n_integration"],
        n_frames=params["n_frames"],
        n_t_plot=params["n_t_plot"],
        interval=params["interval"],
        repeat=params["repeat"],
        save_path=save_path,
        fps=params["fps"],
        dpi=params["dpi"],
        show=(not SAVE_GIF) or SHOW_ANIMATION_AFTER_SAVING,
    )
