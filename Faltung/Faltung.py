import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

safe_gif = False

#####################################
# define functions to convolve here #
#####################################

# def box(x):
#     T = 0.5
#     return (x>-T) * (x<T) * 1.0

# def step(x):
#     return (x > 0) * 1.0

# def biphasic_unitary_step(x):
#     return box(x-0.5)-box(x-1.5)

# def exponential(x):
#     return np.exp(x)

# def triangle(x):
#     # Triangle with width 1
#     y = ((x+1) * step(+1) -  2*x*step(x))*box(1/2*x)
#     return y

# def exponential_decay(x):
#     return np.exp(-x)

# def positive_exponential_decay(x):
#     ''' Exponential function which is 0 for x<0'''
#     return np.exp(-x) * step(x)

# def sinusoid_1_period(x):
#     period = np.pi  # set the period of the sin. could make smaller or larger
#     box_stretch = np.pi  #1 period wide
#     box_shift = box_stretch/2  # The box should be shifted so it starts at 0 instead of -0.5
    
#     y = np.sin(x*(2*np.pi)/period) * box(1/box_stretch*(x-box_shift))
#     return y

# def sinusoid_3_period_phase(x):
#     period = 1/3*np.pi  # set the period of the sine. could make smaller or larger
#     phase_shift = 0  # 180 degree phase shift. could change this
#     box_stretch = np.pi  # the box goes from 0 to pi to select only part of the sine
    
#     box_shift = box_stretch/2  # The box should be shifted so it starts at 0
#     y = np.sin(x*(2*np.pi)/period + phase_shift) * box(1/box_stretch*(x-box_shift))
#     return y

# def sinusoid_3_period_phase_shifted(x):
#     period = 1/3*np.pi  # set the period of the sine. could make smaller or larger
#     phase_shift = np.pi  # 180 degree phase shift. could change this
#     box_stretch = np.pi  # the box goes from 0 to pi to select only part of the sine
    
#     box_shift = box_stretch/2  # The box should be shifted so it starts at 0
#     y = np.sin(x*(2*np.pi)/period + phase_shift) * box(1/box_stretch*(x-box_shift))
#     return y

# def M_1(x):
#     return np.where(np.abs(x) < 1, 1 - np.abs(x), 0)

# def M_k(k,x):
#     if k == 1:
#         return M_1(x)
#     else:
#         return np.convolve(M_1(x), M_k(k-1, x), mode='same') * (x[1]-x[0])
    
# def tut_3_4(x):
#     return 1/(1+x**2)

# def tut_3_4_b(x):
#     return np.exp( -np.abs(x))

# def your_function(x):
#     # quadratic * stretched box
#     y_value = 1.0 * M_k(4,x)
#     return y_value

def parameters():
    return {

        "f_l": 0,
        "f_r": 1,
        # funktion auf dem intervall [f_l, f_r]
        "f": lambda x: 1,

        "g_l": 0,
        "g_r": 1,
        "g": lambda x: 1,

        "n_plot": 1200,
        "n_integration": 5000,
        "interval_label": "[-2, 2]",
    }

# animation der faltung von f und g. x ist der verschiebungsparameter
def animate_convolution(f, g, x, f_l, f_r, g_l, g_r, n_integration):
    L = f_r - f_l  # period length for f
    M = g_r - g_l  # period length for g
    convolution_values = np.zeros_like(x)
    t = np.linspace(f_l, f_r, n_integration, endpoint=False)
    dt = t[1] - t[0]
    ft = f(t)
    gt = g(t)
    for i, shift in enumerate(x):
        # Compute the convolution at this shift
        shifted_gt = np.interp(t - shift, t, gt, left=0, right=0)
        convolution_values[i] = np.sum(ft * shifted_gt) * dt
    return convolution_values

animate_convolution(parameters()["f"], parameters()["g"], np.linspace(-2, 2, parameters()["n_plot"]), parameters()["f_l"], parameters()["f_r"], parameters()["g_l"], parameters()["g_r"], parameters()["n_integration"])


# Optional: save animation
if safe_gif == True:
    from pathlib import Path

    gifs_dir = Path("./gifs")
    gifs_dir.mkdir(exist_ok=True)

    i = 1
    while (gifs_dir / f"faltung_animation_{i}.gif").exists():
        i += 1

    save_path = gifs_dir / f"faltung_animation_{i}.gif"
    print(save_path)
    ani = animate_convolution(parameters()["f"], parameters()["g"], np.linspace(-2, 2, parameters()["n_plot"]), parameters()["f_l"], parameters()["f_r"], parameters()["g_l"], parameters()["g_r"], parameters()["n_integration"], interval=40, save_path=str(save_path), fps=20)

    # ani = animate_fourier_transform(data, interval=40, save_path="fourier_transform_animation.mp4", fps=20)