"""A demo from 2019 plk lans
"""

import sys
import time

import matplotlib.cm as cm
import numpy as np

from vispy import app
from vispy import scene

from vispy.visuals.transforms import MatrixTransform

from vispy.util.transforms import translate
from vispy.util.transforms import rotate

from math import pi
from math import cos
from math import sin

from vispy.geometry.parametric import surface


def klein(u, v):
    if u < pi:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(u) * cos(v)
        z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v)
    else:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + pi)
        z = -8 * sin(u)
    y = -2 * (1 - cos(u) / 2) * sin(v)

    return x / 5, y / 5, z / 5


def add_snake(vis_mask, t, start_time=0, start_loc=0, snake_len=10, direction=1):
    if t >= start_time and t < start_time + 2 * snake_len:
        location_idx = t - start_time
    else:
        return vis_mask

    width = max(min(snake_len - location_idx, location_idx), 0)

    if width == 0:
        return vis_mask

    for idx in range(width):
        vis_mask[
            (start_loc - 1 * direction + direction * location_idx - direction * idx)
            % len(vis_mask)
        ] = True

    return vis_mask


def effect_roll(t, length, start, end):
    vis_mask = [False] * length

    for start_time in range(start, end, 127):
        vis_mask = add_snake(
            vis_mask, t, start_time=start_time, start_loc=0, snake_len=142, direction=1
        )

    return vis_mask


def get_klein_path(n_points):
    """Generate a path along the Klein bottle's tube."""
    u_values = np.linspace(pi, 3 * pi, n_points)
    path_1 = np.array([klein(u, pi) for u in u_values])
    path_2 = np.array([klein(u, 0) for u in u_values])
    path = (path_1 + path_2) / 2
    return np.array(path)


def get_diving_camera(t, start, path):
    t = t - start
    if t >= 0 and t < 200:
        return np.array([0.0, 5.0 - t * 0.010, 0.0]), np.array([0.0, 0.0, 0.0])
    elif t >= 200 and t < 300:
        s = t - 200
        start = np.array([0.0, 3.0, 0.0])
        end = path[0]
        return start + (s / 100) * (end - start), np.array([0.0, 0.0, 0.0])
    elif t >= 300 and t < 500:
        s = t - 300
        return path[s], np.array([0.0, 0.0, 0.0])
    elif t >= 500 and t < 700:
        s = t - 500
        return path[s], np.array([0.0, 0.0, 0.0])
    elif t >= 700 and t < 800:
        s = t - 700
        start = path[0]
        end = np.array([0.0, -5.0, 0.0])
        return start + (s / 100) * (end - start), np.array([0.0, 0.0, 0.0])


class Demo:
    """ """

    def __init__(self):
        """ """
        self.canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.ArcballCamera(
            parent=self.view.scene, distance=5.0, fov=60.0, interactive=False
        )
        self.camera_origin = np.array([0, 5.0, 0.0])

        self.klein_surface = []
        u_space = np.linspace(0, 2 * np.pi, 128)
        for idx in range(len(u_space) - 1):
            umin, umax = u_space[idx], u_space[idx + 1]

            # Add mesh
            vertices, indices = surface(
                klein, urepeat=1, vrepeat=1, umin=umin, umax=umax, ucount=4, vcount=8
            )
            indices = indices.reshape(len(indices) // 3, 3)
            mesh = scene.visuals.Mesh(
                vertices=vertices["position"],
                faces=indices,
                # color="red",
                color=(1,0,0,0.7),
                parent=self.view.scene,
                shading="flat",
            )
            mesh.parent = None
            mesh.transform = MatrixTransform()
            self.klein_surface.append(mesh)

        self.path = get_klein_path(200)
        # self.centerline = scene.visuals.Line(self.path, color='blue', width=5, parent=None)
        # self.centerline.transform = MatrixTransform()

        self.texts = {}

        self.acts = []

        # act 1
        text = scene.visuals.Text(
            "The infamous klein bottle..", parent=None, color="red"
        )
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"text": text, "elapsed": 0, "text_time": 30, "visual_time": 20}
        )

        # act 2
        text = scene.visuals.Text("..may be projected..", parent=None, color="red")
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 30, "visual_time": 20}
        )

        # act 3
        text = scene.visuals.Text("..into three dimensions..", parent=None, color="red")
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 30, "visual_time": 20}
        )

        # act 4
        text = scene.visuals.Text("..however..", parent=None, color="red")
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 30, "visual_time": 20}
        )

        # act 5
        text = scene.visuals.Text("..why should we?", parent=None, color="red")
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 30, "visual_time": 240}
        )

        # act 6
        text = scene.visuals.Text(
            "We live inside space-time.", parent=None, color="red"
        )
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 30, "visual_time": 280}
        )

        # act 7
        text = scene.visuals.Text("We are the klein bottle.", parent=None, color="red")
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 30, "visual_time": 600}
        )

        # act 8
        text = scene.visuals.Text("thx plk s2019 forever", parent=None, color="red")
        text.font_size = 32
        text.pos = (
            self.canvas.physical_size[0] / 2.0,
            self.canvas.physical_size[1] / 2.0,
        )

        self.acts.append(
            {"elapsed": 0, "text": text, "text_time": 50, "visual_time": 50}
        )

        self.timer = app.Timer(interval=0.02, start=True, connect=self.update_scene)
        app.run()

    def move_camera_to(self, location):
        difference = location - self.camera_origin

        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx],
                translate((-difference[0], difference[1], -difference[2])),
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix,
            translate((-difference[0], difference[1], -difference[2])),
        )


    def act_1(self, global_time, t):
        """ """
        self.vis_mask = [True] * self.surface_size

        start = time.time()
        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(3.0 * t, (0.0, 0.0, 1.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(3.0 * t, (0.0, 0.0, 1.0))
        )


    def act_2(self, global_time, t):
        """ """
        t = t + 20
        self.vis_mask = [True] * self.surface_size

        start = time.time()
        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(3.0 * t, (0.0, 0.0, 1.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(3.0 * t, (0.0, 0.0, 1.0))
        )

    def act_3(self, global_time, t):
        """ """
        t = t + 40
        self.vis_mask = [True] * self.surface_size

        start = time.time()
        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(3.0 * t, (0.0, 0.0, 1.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(3.0 * t, (0.0, 0.0, 1.0))
        )


    def act_4(self, global_time, t):
        """ """
        t = t + 60
        self.vis_mask = [True] * self.surface_size

        start = time.time()
        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(3.0 * t, (0.0, 0.0, 1.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(3.0 * t, (0.0, 0.0, 1.0))
        )


    def act_5(self, global_time, t):
        """ """
        self.vis_mask = effect_roll(t, self.surface_size, start=0, end=260)

        start = time.time()
        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(1.0 * t, (0.0, 0.0, 1.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(1.0 * t, (0.0, 0.0, 1.0))
        )

    def act_6(self, global_time, t):
        """ """
        t = t + 260

        vis_mask = [False] * self.surface_size

        for start_time in range(260, 540, 32):
            vis_mask = add_snake(
                vis_mask,
                t,
                start_time=start_time,
                start_loc=20,
                snake_len=66,
                direction=1,
            )
            vis_mask = add_snake(
                vis_mask,
                t,
                start_time=start_time,
                start_loc=20,
                snake_len=66,
                direction=-1,
            )

        self.vis_mask = vis_mask

        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(270, (0.0, 0.0, 1.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(270, (0.0, 0.0, 1.0))
        )


    def act_7(self, global_time, t):
        """ """
        t = t + 540

        self.vis_mask = effect_roll(t, self.surface_size, start=540, end=1040)
        # self.vis_mask = [True] * self.surface_size

        for idx, mesh in enumerate(self.klein_surface):
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(270, (0.0, 0.0, 1.0))
            )
            self.transformation_matrix[idx] = np.dot(
                self.transformation_matrix[idx], rotate(90, (1.0, 0.0, 0.0))
            )

        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(270, (0.0, 0.0, 1.0))
        )
        self.centerline_transformation_matrix = np.dot(
            self.centerline_transformation_matrix, rotate(90, (1.0, 0.0, 0.0))
        )

        # This tries to replicate the above transform, but does not actually work..
        transformed_path = np.dot(
            self.centerline_transformation_matrix,
            np.concatenate((self.path, np.ones((self.path.shape[0], 1))), axis=1).T
        )[:-1, :].T

        # and for this reason, try hot fix for lack of wisdom. This is ugly.
        transformed_path = transformed_path[:, [1, 0, 2]]
        transformed_path = transformed_path[:, [2, 1, 0]]
        transformed_path[:, 2] = -transformed_path[:, 2]
        transformed_path[:, 1] = -transformed_path[:, 1]
        transformed_path = transformed_path[::-1, :]

        # from PyQt5.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace()

        trans, rotat = get_diving_camera(t, start=540, path=transformed_path)

        self.move_camera_to(trans)


    def act_8(self, global_time, t):
        """ """
        self.vis_mask = [False] * self.surface_size

    def update_scene(self, event):
        """ """

        self.surface_size = len(self.klein_surface)
        self.vis_mask = [False] * self.surface_size
        self.transformation_matrix = [np.eye(4) for idx in range(self.surface_size)]
        self.centerline_transformation_matrix = np.eye(4)

        for act_idx, act in enumerate(self.acts):
            if act["elapsed"] == act["text_time"] + act["visual_time"]:
                if act_idx == len(self.acts) - 1:
                    app.quit()
                    self.timer.stop()
                    break
                continue

            if act["elapsed"] < act["text_time"]:
                act["text"].parent = self.canvas.scene
                self.vis_mask = [False] * self.surface_size
            else:
                act["text"].parent = None
                getattr(self, "act_" + str(act_idx + 1))(
                    event.iteration, act["elapsed"] - act["text_time"]
                )

            act["elapsed"] += 1
            break

        for idx in range(self.surface_size):
            if self.vis_mask[idx]:
                self.klein_surface[idx].parent = self.view.scene
                self.klein_surface[idx].transform.matrix = self.transformation_matrix[
                    idx
                ]
            else:
                self.klein_surface[idx].parent = None

        # self.centerline.transform.matrix = self.centerline_transformation_matrix
        # self.centerline.parent = self.view.scene

        self.canvas.update()


if __name__ == "__main__" and sys.flags.interactive == 0:
    demo = Demo()
