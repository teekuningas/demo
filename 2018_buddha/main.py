"""A demo from 2018: The Buddha getting her satori.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import geometric_transform


def shift_func(coords, angle, Rz, cent_x, cent_y):
    X = (coords[0] - cent_x) * 1.0
    Y = (coords[1] - cent_y) * 1.0

    x = (2 * X) / (1.0 + X**2 + Y**2)
    y = (2 * Y) / (1.0 + X**2 + Y**2)
    z = (-1 + X**2 + Y**2) / (1 + X**2 + Y**2)

    x, y, z = np.dot(Rz, [x, y, z])

    X = x / (1 - z)
    Y = y / (1 - z)

    return X + cent_x, Y + cent_y, coords[2]


def get_frame(angle, scale, img, cent_x, cent_y):
    Rz = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )

    newimg = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            coords = (i, j, None)
            new_coords = shift_func(coords, angle, Rz, cent_x, cent_y)
            try:
                newimg[int(new_coords[0]), int(new_coords[1]), :] = img[i, j, :]
            except IndexError:  # More specific exception
                pass

    return newimg


img = np.fliplr(plt.imread("buddha_scaled.jpg"))
cent_x = img.shape[0] / 2
cent_y = img.shape[1] / 2

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title("Buddha")

for angle in np.exp(np.arange(0, 5, 0.004)) - 1:
    if not plt.fignum_exists(fig.number):
        break
    plt.cla()
    ax.set_xticks([])
    ax.set_yticks([])

    scale = angle
    if np.isclose(angle, 0):
        ax.imshow(img)
    else:
        ax.imshow(get_frame(angle, scale, img, cent_x, cent_y))
    plt.draw()
    plt.pause(0.1)

plt.show()
