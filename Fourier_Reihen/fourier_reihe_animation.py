import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

safe_gif = False

def parameters():
    return {
        "l": -1,
        "r": 1,
        "f": lambda x: x,
        "n_terms": 50,
        "n_plot": 1200,
        "n_integration": 5000,
        "interval_label": "[-2, 2]",
    }


def compute_fourier_coefficients(f, l, r, n_terms, n_integration=5000):
    """
    Compute Fourier coefficients once and cache them.

    We use the real Fourier series on [l, r] with period L = r-l:
        S_N(x) = a0/2 + sum_{n=1}^N [a_n cos(2πn(x-l)/L) + b_n sin(2πn(x-l)/L)]

    Important:
    - a0 = (2/L) ∫_l^r f(t) dt
    - a_n = (2/L) ∫_l^r f(t) cos(2πn(t-l)/L) dt
    - b_n = (2/L) ∫_l^r f(t) sin(2πn(t-l)/L) dt
    """
    L = r - l
    t = np.linspace(l, r, n_integration, endpoint=False)
    ft = f(t)
    phase_t = 2 * np.pi * (t - l) / L

    a0 = (2.0 / L) * np.trapezoid(ft, t)
    a = np.zeros(n_terms + 1)
    b = np.zeros(n_terms + 1)

    for n in range(1, n_terms + 1):
        a[n] = (2.0 / L) * np.trapezoid(ft * np.cos(n * phase_t), t)
        b[n] = (2.0 / L) * np.trapezoid(ft * np.sin(n * phase_t), t)

    return a0, a, b


def evaluate_fourier_series(x, coeffs, l, r, n_terms=None):
    a0, a, b = coeffs
    max_terms = len(a) - 1

    if n_terms is None:
        n_terms = max_terms
    n_terms = min(n_terms, max_terms)

    L = r - l
    phase_x = 2 * np.pi * (x - l) / L

    s = np.full_like(x, a0 / 2.0, dtype=float)
    for n in range(1, n_terms + 1):
        s += a[n] * np.cos(n * phase_x) + b[n] * np.sin(n * phase_x)
    return s


def build_partial_sums(x, coeffs, l, r):
    """
    Build all partial sums incrementally so the animation does not recompute
    old terms over and over again.
    """
    a0, a, b = coeffs
    n_terms = len(a) - 1
    L = r - l
    phase_x = 2 * np.pi * (x - l) / L

    partial_sums = []
    current = np.full_like(x, a0 / 2.0, dtype=float)
    partial_sums.append(current.copy())  # 0 modes, only constant term

    for n in range(1, n_terms + 1):
        current = current + a[n] * np.cos(n * phase_x) + b[n] * np.sin(n * phase_x)
        partial_sums.append(current.copy())

    return partial_sums



def plot_selected_partial_sums(x, f_x, partial_sums, selection=(0, 1, 2, 5, 10, 20, 50)):
    plt.figure(figsize=(12, 6))
    plt.plot(x, f_x, linewidth=2, label="Original function")

    for n in selection:
        if n < len(partial_sums):
            plt.plot(x, partial_sums[n], label=f"N = {n}")

    plt.title("Fourier series approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def animate_fourier_series(x, f_x, partial_sums, interval=120, repeat=True):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, f_x, linewidth=2, label="Original function")
    approx_line, = ax.plot(x, partial_sums[0], linewidth=2, label="Fourier approximation")

    y_min = min(np.min(f_x), np.min(partial_sums[-1]))
    y_max = max(np.max(f_x), np.max(partial_sums[-1]))
    margin = 0.1 * (y_max - y_min + 1e-12)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    def update(frame):
        approx_line.set_ydata(partial_sums[frame])
        ax.set_title(f"Fourier series approximation with N = {frame} modes")
        return (approx_line,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(partial_sums),
        interval=interval,
        blit=True,
        repeat=repeat,
    )

    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    params = parameters()
    l = params["l"]
    r = params["r"]
    f = params["f"]
    n_terms = params["n_terms"]
    n_plot = params["n_plot"]
    n_integration = params["n_integration"]

    x = np.linspace(l, r, n_plot)
    f_x = f(x)

    coeffs = compute_fourier_coefficients(
        f=f,
        l=l,
        r=r,
        n_terms=n_terms,
        n_integration=n_integration,
    )

    partial_sums = build_partial_sums(x, coeffs, l, r)

    # Static overview plot
    plot_selected_partial_sums(x, f_x, partial_sums, selection=(0, 1, 2, 5, 10, 20, 50))

    # Animation
    ani = animate_fourier_series(x, f_x, partial_sums, interval=120)

    # Optional: save animation
    if safe_gif == True:
        from pathlib import Path

        gifs_dir = Path("./gifs")
        gifs_dir.mkdir(exist_ok=True)

        i = 1
        while (gifs_dir / f"fourier_series_animation_{i}.gif").exists():
            i += 1

        save_path = gifs_dir / f"fourier_series_animation_{i}.gif"
        print(save_path)
        ani = animate_fourier_series(x, f_x, partial_sums, interval=40, save_path=str(save_path), fps=20)
        
        # ani = animate_fourier_transform(data, interval=40, save_path="fourier_transform_animation.mp4", fps=20)

