#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for Key estimation notebook
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseEvent

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))


class InteractiveCircleOfFifths(object):
    """
    Interactive visualization for the circle of fifths.

    Parameters
    ----------
    start_key : str
        Key at 12 o'clock in the Circle of Fifths
    radius: float, optional
        Radius of the circle for major keys
    circle_radius: float, optional
        Radius of the small circles for key names
    """

    def __init__(
        self,
        start_key: str = "C",
        radius: float = 5,
        circle_radius: float = 0.5,
    ) -> None:
        self.start_key = start_key
        self.radius = radius
        self.circle_radius = circle_radius
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.axis("off")
        self.clicked_points = []
        self.selection_markers = []
        self.major_keys = []
        self.minor_keys = []
        self.draw_clock()
        self.connect_events()

    def draw_clock(self) -> None:
        # Define the major and minor keys
        major_keys = [
            "C",
            "G",
            "D",
            "A",
            "E",
            "B",
            "F#",
            "Db",
            "Ab",
            "Eb",
            "Bb",
            "F",
        ]
        minor_keys = [
            "Am",
            "Em",
            "Bm",
            "F#m",
            "C#m",
            "G#m",
            "D#m",
            "A#m",
            "Fm",
            "Cm",
            "Gm",
            "Dm",
        ]

        # Rotate the lists to start from the specified key
        if self.start_key in major_keys:
            start_index = major_keys.index(self.start_key)
        elif self.start_key.lower() in minor_keys:
            start_index = minor_keys.index(self.start_key.lower())
        else:
            raise ValueError("Invalid key. Please enter a valid major or minor key.")

        major_keys = major_keys[start_index:] + major_keys[:start_index]
        minor_keys = minor_keys[start_index:] + minor_keys[:start_index]

        # Draw the major keys
        for i in range(12):
            angle = 2 * np.pi * i / 12
            x = self.radius * np.sin(angle) * 0.88
            y = self.radius * np.cos(angle) * 0.88
            hour_circle = plt.Circle(
                (x, y),
                self.circle_radius,
                edgecolor="black",
                facecolor="none",
            )
            self.ax.add_patch(hour_circle)
            self.major_keys.append(hour_circle)

        for i in range(12):
            angle = np.pi * i / 6
            x = self.radius * np.sin(angle) * 0.65
            y = self.radius * np.cos(angle) * 0.65
            hour_circle = plt.Circle(
                (x, y),
                self.circle_radius,
                edgecolor="black",
                facecolor="none",
            )
            self.ax.add_patch(hour_circle)
            self.minor_keys.append(hour_circle)

        # Set the aspect ratio of the plot to be equal
        self.ax.set_aspect("equal", "box")
        self.ax.set_xlim([-(self.radius + 1), self.radius + 1])
        self.ax.set_ylim([-(self.radius + 1), self.radius + 1])

        # Place the text for major and minor keys
        for i, (major, minor) in enumerate(zip(major_keys, minor_keys)):
            angle = -i * 2 * np.pi / 12 + np.pi / 2  # Rotate to start at 12 o'clock
            x_major = 0.88 * np.cos(angle) * self.radius
            y_major = 0.88 * np.sin(angle) * self.radius
            x_minor = 0.65 * np.cos(angle) * self.radius
            y_minor = 0.65 * np.sin(angle) * self.radius
            self.ax.text(
                x_major,
                y_major,
                major,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )
            self.ax.text(
                x_minor,
                y_minor,
                minor,
                ha="center",
                va="center",
                fontsize=10,
            )

    def connect_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event: MouseEvent):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        # Left mouse button
        if event.button == 1:
            # all circles except for the central circle
            for circle in self.ax.patches:
                center_x, center_y = circle.center
                radius = circle.radius
                is_minor = False
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                    is_minor = circle in self.minor_keys
                    self.clicked_points.append((center_x, center_y, is_minor))
                    marker = plt.plot(center_x, center_y, "ro")[0]
                    self.selection_markers.append(marker)
                    plt.draw()

                    if len(self.clicked_points) == 2:
                        self.calculate_tonal_distance()

                    if len(self.clicked_points) > 2:
                        self.clicked_points = []
                        for marker in self.selection_markers:
                            marker.remove()
                        self.selection_markers = []
        # Right mouse button
        elif event.button == 3:
            self.deselect_point(x, y)

    def deselect_point(self, x: float, y: float) -> None:
        if not self.clicked_points:
            return
        closest_point_index = np.argmin(
            [np.hypot(x - cx, y - cy) for cx, cy, _ in self.clicked_points]
        )
        del self.clicked_points[closest_point_index]
        self.selection_markers[closest_point_index].remove()
        del self.selection_markers[closest_point_index]
        plt.draw()

    def calculate_tonal_distance(self) -> None:
        (x1, y1, m1), (x2, y2, m2) = self.clicked_points

        # Vectors from origin to points
        v1 = np.array([x1, y1])
        v2 = np.array([x2, y2])

        # Dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Calculate the angle
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        t1 = "minor" if m1 else "major"
        t2 = "minor" if m2 else "major"

        # Compute tonal distance
        tonal_distance = 6 * angle / np.pi + 1 if (m1 != m2) else 6 * angle / np.pi
        print(f"The tonal distance is {tonal_distance:.2f}")


def draw_circle_of_fifths(start_key: str = "C") -> None:
    # Define the major and minor keys
    major_keys = ["C", "G", "D", "A", "E", "B", "F♯", "D♭", "A♭", "E♭", "B♭", "F"]
    minor_keys = ["a", "e", "b", "f♯", "c♯", "g♯", "d♯", "b♭", "f", "c", "g", "d"]

    # Rotate the lists to start from the specified key
    if start_key in major_keys:
        start_index = major_keys.index(start_key)
    elif start_key.lower() in minor_keys:
        start_index = minor_keys.index(start_key.lower())
    else:
        raise ValueError("Invalid key. Please enter a valid major or minor key.")

    major_keys = major_keys[start_index:] + major_keys[:start_index]
    minor_keys = minor_keys[start_index:] + minor_keys[:start_index]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    # Draw the circles
    circle_outer = plt.Circle(
        (0, 0),
        1,
        color="firebrick",
        edgecolor="black",
        linewidth=1.5,
        alpha=1.0,
    )
    circle_inner = plt.Circle(
        (0, 0),
        0.8,
        color="gray",
        edgecolor="black",
        linewidth=1.5,
        alpha=1.0,
    )
    circle_center = plt.Circle(
        (0, 0),
        0.6,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        alpha=1.0,
    )
    ax.add_patch(circle_outer)
    ax.add_patch(circle_inner)
    ax.add_patch(circle_center)

    # Place the text for major and minor keys
    for i, (major, minor) in enumerate(zip(major_keys, minor_keys)):
        angle = -i * 2 * np.pi / 12 + np.pi / 2  # Rotate to start at 12 o'clock
        x_major = 0.88 * np.cos(angle)
        y_major = 0.88 * np.sin(angle)
        x_minor = 0.72 * np.cos(angle)
        y_minor = 0.72 * np.sin(angle)
        ax.text(
            x_major,
            y_major,
            major,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
        ax.text(
            x_minor,
            y_minor,
            minor,
            ha="center",
            va="center",
            fontsize=10,
        )
