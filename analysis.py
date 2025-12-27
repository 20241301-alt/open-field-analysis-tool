import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
from matplotlib.path import Path

# ============================================================
# Configuration /
# ============================================================

# Actual size of the arena/box in cm
REAL_BOX_SIZE_CM = 

# Threshold for filtering low-confidence body parts
LIKELIHOOD_THRESHOLD = 

# Max frames for interpolation (limit for filling NaNs)
INTERPOLATION_LIMIT = 

# ============================================================
# 1. GUI Input / 
# ============================================================
root = tk.Tk()
root.withdraw() 

print("Please select the folder containing video files and Excel data...")
dir_path = filedialog.askdirectory(title="Select Data Directory")

if not dir_path:
    print("No directory selected. Exiting.")
    sys.exit()

target_part = simpledialog.askstring(
    "Body Part", 
    "Enter the body part name to analyze (e.g., head, snout, body):", 
    initialvalue="head"
)

if not target_part:
    print("No body part specified. Exiting.")
    sys.exit()

# Folder to save results
figure_dir = os.path.join(dir_path, "Analysis_Results")
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

excel_files = [
    f for f in os.listdir(dir_path) if f.endswith(".xlsx") and not f.startswith("._")
]

# ============================================================
# Helper Functions / 
# ============================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps

def get_corner_coordinates_manually(image_data):
    plt.figure(figsize=(10, 8))
    plt.imshow(image_data)
    plt.title(f"Click 4 corners (TL -> TR -> BL -> BR)\nReal Size: {REAL_BOX_SIZE_CM}cm")
    plt.xlabel("Click 4 points, then CLOSE the window to continue.")
    
    points = plt.ginput(4, timeout=0)
    plt.close()

    if len(points) != 4:
        return None
    return points

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ============================================================
# Main Analysis Loop / 
# ============================================================
all_results = []

print(f"Found {len(excel_files)} files. Starting analysis...")

col_x = f"{target_part}_x"
col_y = f"{target_part}_y"
col_like = f"{target_part}_likelihood"

for excel_file in excel_files:
    base_name = os.path.splitext(os.path.basename(excel_file))[0]
    video_base_name = base_name.split("DLC")[0]
    
    video_path = None
    for ext in [".mov", ".mp4", ".avi"]:
        temp_path = os.path.join(dir_path, f"{video_base_name}{ext}")
        if os.path.exists(temp_path):
            video_path = temp_path
            break
    
    if video_path is None:
        continue

    try:
        print(f"Processing: {excel_file}")
        first_frame, current_fps = get_video_info(video_path)
        if first_frame is None: continue

        corner_coords = get_corner_coordinates_manually(first_frame)
        if corner_coords is None: continue

        topL, topR, bottomL, bottomR = corner_coords

        # Calibration
        dists = [
            calculate_distance(topL, topR),
            calculate_distance(bottomL, bottomR),
            calculate_distance(topL, bottomL),
            calculate_distance(topR, bottomR)
        ]
        cm_per_pixel = REAL_BOX_SIZE_CM / np.mean(dists)

        # Region settings
        x_center = sum(p[0] for p in corner_coords) / 4
        y_center = sum(p[1] for p in corner_coords) / 4
        box_width_px = np.mean(dists[:2])
        
        # Load Data
        df = pd.read_excel(os.path.join(dir_path, excel_file), header=[1, 2], engine="openpyxl")
        df.columns = [f"{i}_{j}" for i, j in df.columns]

        # Filter & Interpolate
        df.loc[df[col_like] < LIKELIHOOD_THRESHOLD, [col_x, col_y]] = np.nan
        df[col_x] = df[col_x].interpolate(limit=INTERPOLATION_LIMIT)
        df[col_y] = df[col_y].interpolate(limit=INTERPOLATION_LIMIT)
        df = df.dropna(subset=[col_x, col_y])

        # ROI Mask
        polygon = Path([topL, topR, bottomR, bottomL])
        df = df[polygon.contains_points(df[[col_x, col_y]])]

        x_coords, y_coords = df[col_x].values, df[col_y].values
        step_dists_px = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        
        total_dist_cm = np.sum(step_dists_px) * cm_per_pixel
        velocity = total_dist_cm / (len(x_coords) / current_fps)

        # Zone Analysis (30% & 50%)
        def analyze_zone(size_ratio):
            side = box_width_px * size_ratio
            in_zone = (np.abs(x_coords - x_center) <= side/2) & (np.abs(y_coords - y_center) <= side/2)
            dist_cm = np.sum(step_dists_px[in_zone[:-1]]) * cm_per_pixel
            time_sec = np.sum(in_zone) / current_fps
            return round(dist_cm, 2), round(time_sec, 2)

        d30, t30 = analyze_zone(0.3)
        d50, t50 = analyze_zone(0.5)

        all_results.append({
            "File Name": excel_file, "Part": target_part, "FPS": round(current_fps, 2),
            "Total Distance (cm)": round(total_dist_cm, 2), "Velocity (cm/s)": round(velocity, 2),
            "Dist 30% (cm)": d30, "Time 30% (s)": t30, "Dist 50% (cm)": d50, "Time 50% (s)": t50
        })

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.plot(x_coords, y_coords, color="gray", alpha=0.5, lw=1)
        plt.gca().add_patch(plt.Polygon(corner_coords, fill=None, edgecolor="red", ls="--", label="ROI"))
        plt.title(f"Trajectory: {base_name}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(figure_dir, f"{base_name}.png"))
        plt.close()

    except Exception as e:
        print(f"Error in {excel_file}: {e}")

if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_df.to_excel(os.path.join(figure_dir, "summary.xlsx"), index=False)
    print("Analysis finished successfully.")
