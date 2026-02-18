from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw


def save_accuracy_line_plot(
    *,
    rounds: Sequence[int],
    accuracies: Sequence[float],
    output_path: Path,
    title: str = "Test Accuracy vs Communication Rounds",
) -> None:
    if len(rounds) != len(accuracies):
        raise ValueError("rounds and accuracies must have the same length.")
    if not rounds:
        raise ValueError("rounds must not be empty.")

    width = 960
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 65
    margin_bottom = 80

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if plot_width <= 0 or plot_height <= 0:
        raise ValueError("Invalid plotting canvas dimensions.")

    x_min = min(rounds)
    x_max = max(rounds)
    if x_min == x_max:
        x_max = x_min + 1

    y_min = 0.0
    y_max = max(1.0, max(float(value) for value in accuracies))
    if y_max == y_min:
        y_max = y_min + 1.0

    def x_to_px(value: float) -> float:
        return margin_left + ((value - x_min) / float(x_max - x_min)) * plot_width

    def y_to_px(value: float) -> float:
        return margin_top + (1.0 - ((value - y_min) / float(y_max - y_min))) * plot_height

    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # Axes.
    draw.line(
        [(margin_left, margin_top), (margin_left, margin_top + plot_height)],
        fill="black",
        width=2,
    )
    draw.line(
        [
            (margin_left, margin_top + plot_height),
            (margin_left + plot_width, margin_top + plot_height),
        ],
        fill="black",
        width=2,
    )

    # Horizontal grid lines and y-axis labels.
    y_ticks = 5
    for index in range(y_ticks + 1):
        fraction = index / float(y_ticks)
        y_value = y_min + (y_max - y_min) * fraction
        y_pixel = y_to_px(y_value)
        draw.line(
            [(margin_left, y_pixel), (margin_left + plot_width, y_pixel)],
            fill="#dddddd",
            width=1,
        )
        draw.text((16, y_pixel - 8), f"{y_value:.2f}", fill="black")

    # Round labels on x-axis.
    x_ticks = min(10, max(2, len(rounds)))
    for index in range(x_ticks):
        fraction = index / float(x_ticks - 1)
        x_value = x_min + (x_max - x_min) * fraction
        x_pixel = x_to_px(x_value)
        draw.line(
            [(x_pixel, margin_top + plot_height), (x_pixel, margin_top + plot_height + 6)],
            fill="black",
            width=1,
        )
        draw.text(
            (x_pixel - 10, margin_top + plot_height + 10),
            f"{int(round(x_value))}",
            fill="black",
        )

    points = [
        (x_to_px(float(round_value)), y_to_px(float(accuracy_value)))
        for round_value, accuracy_value in zip(rounds, accuracies, strict=True)
    ]
    if len(points) > 1:
        draw.line(points, fill="#1f77b4", width=3)
    for point_x, point_y in points:
        draw.ellipse(
            [(point_x - 3, point_y - 3), (point_x + 3, point_y + 3)],
            fill="#1f77b4",
            outline="#1f77b4",
        )

    draw.text((margin_left, 18), title, fill="black")
    draw.text((margin_left + plot_width // 3, height - 32), "Communication Round", fill="black")
    draw.text((16, 28), "Accuracy", fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
