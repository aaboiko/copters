import numpy as np

func_static_zero = lambda x, y, z, t: np.zeros(3)
#motion_func_linear = lambda t: np.array([])

external_force_functions = {
    "static_zero": func_static_zero
}