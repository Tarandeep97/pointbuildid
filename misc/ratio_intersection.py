import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm 

output_dir = "./output"
input_dir = "/media/ritesh/Partition4/Taran/mask2points/"

# This is sampled mask path, original masks at mask2points/AerialImageDatasetWVal/train/gt
img_path = os.path.join(input_dir, "AerialImageDatasetWVal/train/gt")
img_ids = glob.glob(os.path.join(img_path, "*.tif")) #*tif for original masks
# print(img_path)

def calculate_mask_to_outside_fraction(cnts, msk, radius, centroid_outside_counts):
    if not cnts:  
        return 0, 0, centroid_outside_counts
    ratios = []
    buildings = len(cnts)
    for cnt in cnts:
        #find centroid of contour
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue  
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])


        #check if centroid is outside the contour
        if cv2.pointPolygonTest(cnt, (cx, cy), False) <= 0:  
            centroid_outside_counts += 1
            buildings-=1
            continue

        #create circle mask
        circle_mask = np.zeros_like(msk, dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

        #create contour mask
        contour_mask = np.zeros_like(msk, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        #create and compute intersection mask
        intersection_mask = cv2.bitwise_and(circle_mask, contour_mask)
        area_intersection = np.count_nonzero(intersection_mask)

        #compute area of circle
        area_circle = np.count_nonzero(circle_mask)

        if area_circle == 0:
            ratio = 0
        else:
            ratio = area_intersection / area_circle
        ratios.append(ratio)

    if not ratios:
        return 0, 0, centroid_outside_counts
    return sum(ratios), buildings, centroid_outside_counts


results = []
radii = range(1, 100)
total_centroids = 0
total_centroids_outside = 0

for radius in tqdm(radii, desc="Processing each radius", leave=False):
    radius_fractions = []
    centroid_outside_counts = 0
    centroids_in_radius = 0 
    tot_buildings = 0
    for img_id in tqdm(img_ids, desc=f"Radius {radius}", leave=False):
        msk = cv2.imread(img_id, cv2.IMREAD_GRAYSCALE)
        contours_data = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        fraction, buildings, centroid_outside_counts = calculate_mask_to_outside_fraction(cnts, msk, radius, centroid_outside_counts)
        centroids_in_radius += len(cnts)
        tot_buildings += buildings
        radius_fractions.append(fraction)
    avg_fraction = sum(radius_fractions)/tot_buildings if radius_fractions else 0
    results.append({"radius": radius, "fraction": avg_fraction})
    total_centroids_outside = centroid_outside_counts
    total_centroids = centroids_in_radius


print(f"Total number of contours: {total_centroids}")
print(f"Total number of centroids outside: {total_centroids_outside}")
print(f"Percentage of centroids outside: {100 * total_centroids_outside / total_centroids:.2f}%")

results_df = pd.DataFrame(results)

plt.rcParams.update({
    'font.size': 17,  
    'figure.figsize': (8, 6),
    'savefig.dpi': 300,
    'text.usetex': True, 
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  
    'lines.linewidth': 4,  
    'lines.markersize': 6,
    'axes.labelsize': 18, 
    'axes.titlesize': 16, 
    'legend.fontsize': 16, 
    'xtick.labelsize': 16, 
    'ytick.labelsize': 16 
})

plt.plot(results_df['radius'], results_df['fraction'], color='red')
plt.xlabel('Radius $r$ (pixels)')
plt.ylabel(r'$|\mathcal{M} \cap \mathcal{B}(r)|/|\mathcal{B}(r)|$')
plt.grid(True)

plt.xticks(range(1, 100, 10))
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "ratio_intersection.pdf"))
plt.show()
