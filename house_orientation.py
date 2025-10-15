
import argparse
import math
import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------- Math helpers ----------

def normalize_angle_deg(a: float) -> float:
    """Normalize angle to [0,360)."""
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def bearing_from_vector(dx: float, dy: float) -> float:
    """
    Convert image-space vector (dx, dy) to compass bearing in degrees, where:
      - 0° = North (up)
      - 90° = East (right)
      - 180° = South (down)
      - 270° = West (left)
    Image y increases downward, so bearing = atan2(dx, -dy).
    """
    rad = math.atan2(dx, -dy)
    deg = math.degrees(rad)
    return normalize_angle_deg(deg)

def compass_8(bearing_deg: float) -> str:
    """Map a bearing to the nearest 8-point compass direction."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing_deg + 22.5) // 45) % 8
    return dirs[idx]

def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)

def subtract(p1, p0):
    return (p1[0]-p0[0], p1[1]-p0[1])

def dot(u, v):
    return u[0]*v[0] + u[1]*v[1]

def perp(v):
    """Rotate vector 90° CCW in image coords (x right, y down)."""
    return (-v[1], v[0])

def scale(v, s):
    return (v[0]*s, v[1]*s)

def unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0]/n, v[1]/n) if n != 0 else (0.0, 0.0)

# ---------- Interaction helpers ----------

def get_points(ax, n: int, prompt: str):
    print(prompt)
    pts = plt.ginput(n, timeout=-1)
    if len(pts) != n:
        raise RuntimeError(f"Expected {n} points, got {len(pts)}.")
    # draw points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o", markersize=6)
    for i, p in enumerate(pts):
        ax.text(p[0]+3, p[1]+3, f"{i+1}", fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6))
    ax.figure.canvas.draw()
    return pts

def annotate_arrow(ax, p0, p1, label: str, color="yellow"):
    ax.annotate(label, xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="->", lw=2, color=color),
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))

# ---------- Core workflows ----------

def workflow_vector(ax):
    """
    Fast 2-click method:
      1) Click the front of the house (where it faces the street).
      2) Click a point toward/at the street centerline in front of the house.
    """
    pts = get_points(ax, 2, "Click 1) the front of the house; 2) a point toward the street centerline.")
    p0, p1 = pts[0], pts[1]
    v = subtract(p1, p0)  # vector from house front toward street
    b = bearing_from_vector(v[0], v[1])
    c = compass_8(b)
    annotate_arrow(ax, p0, p1, f"{c} ({b:.1f}°)")
    return b, c

def workflow_frontage(ax):
    """
    Robust 4-click method:
      1-2) Click two points along the HOUSE FRONTAGE LINE (along the facade).
      3-4) Click two points along the STREET (centerline or kerb) in front of the house.
    We take the frontage normal and choose the side that points toward the street.
    """
    house_pts = get_points(ax, 2, "Click 1-2) two points along the house FRONTAGE (along the facade).")
    street_pts = get_points(ax, 2, "Click 3-4) two points along the STREET in front of the house.")

    # Frontage direction vector (along the facade)
    f = subtract(house_pts[1], house_pts[0])
    f_u = unit(f)

    # Normals to the frontage: n_left and n_right (in image coords)
    n_left = unit(perp(f_u))          # 90° CCW from frontage
    n_right = unit(scale(n_left, -1)) # opposite normal

    # Vector from frontage midpoint to street midpoint
    fm = midpoint(house_pts[0], house_pts[1])
    sm = midpoint(street_pts[0], street_pts[1])
    to_street = unit(subtract(sm, fm))

    # Pick the frontage normal that points more toward the street (max dot product)
    n_face = n_left if dot(n_left, to_street) >= dot(n_right, to_street) else n_right

    # Bearing of the chosen normal
    b = bearing_from_vector(n_face[0], n_face[1])
    c = compass_8(b)

    # Annotate
    # Draw frontage line
    ax.plot([house_pts[0][0], house_pts[1][0]], [house_pts[0][1], house_pts[1][1]], "-", lw=2, color="cyan")
    # Draw street line
    ax.plot([street_pts[0][0], street_pts[1][0]], [street_pts[0][1], street_pts[1][1]], "-", lw=2, color="orange")
    # Draw facing arrow from frontage midpoint
    tip = (fm[0] + n_face[0]*60, fm[1] + n_face[1]*60)
    annotate_arrow(ax, fm, tip, f"{c} ({b:.1f}°)")
    return b, c

def main():
    parser = argparse.ArgumentParser(description="Estimate house orientation from a map screenshot.")
    parser.add_argument("image", help="Path to screenshot (PNG/JPG). North must be up.")
    parser.add_argument("--mode", choices=["vector", "frontage"], default="vector",
                        help="vector: 2 clicks (front -> street). frontage: 4 clicks (frontage and street lines).")
    parser.add_argument("-o", "--output", default=None, help="Output annotated PNG path (default: <image>_annotated.png).")
    args = parser.parse_args()

    img = mpimg.imread(args.image)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Click according to mode: {args.mode.upper()}")
    ax.axis("off")

    if args.mode == "vector":
        bearing, compass = workflow_vector(ax)
    else:
        bearing, compass = workflow_frontage(ax)

    # Show text box with result
    txt = f"Facing: {compass} ({bearing:.1f}°)"
    ax.text(10, 20, txt, fontsize=12, color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="none", alpha=0.6))

    # Save annotated output
    base, _ = os.path.splitext(args.image)
    out_path = args.output or f"{base}_annotated.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(txt)
    print(f"Saved annotated image to: {out_path}")

    # Also show interactively so you can visually confirm
    plt.show()

if __name__ == "__main__":
    main()