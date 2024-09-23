import torch
import numpy as np
import cv2
import scipy
from skimage import morphology as morph
from skimage.draw import line_aa
from skimage.segmentation import find_boundaries
from skimage.measure import label as label_fn
from shapely.geometry import Point, box, Polygon
from matplotlib.path import Path
from scipy.spatial import distance
from scipy.spatial import Voronoi
from scipy.spatial import distance as sp_distance


def get_blobs(probs,p=0.5):
    probs = probs.squeeze()
    h, w = probs.shape
 
    pred_mask = (probs>p).astype('uint8')
    blobs = np.zeros((h, w), int)

    blobs = morph.label(pred_mask == 1)

    return blobs

def find_centroid_pixels(image):    
    non_zero_pixels = non_zero_pixel_positions(image)
    return non_zero_pixels   

def non_zero_pixel_positions(image):
    non_zero_pixels = np.array(np.where(image > 0)).T
    return non_zero_pixels

def compute_distances(point, pixel_positions):
    distances = []
    for j in range(0, len(pixel_positions)):
        distance = euclidean(point, pixel_positions[j])
        distances.append(distance)
        
    return distances

def euclidean(point1, point2):
    return distance.euclidean(point1, point2)
  
def voronoi_finite_polygons_2d(vor, radius=None):
    
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
        
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

   
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a Voronoi region bounded by the radius
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1] 
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

 
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
    
    
def huber_loss(error, delta=1.0):
    is_small_error = np.abs(error) <= delta
    
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    
    loss = np.where(is_small_error, squared_loss, linear_loss)
    return np.mean(loss)


def bb_huberloss(points, probs, bb_dim = (10,10)):
    pred = find_centroid_pixels(probs)
    gt = non_zero_pixel_positions(points)
    
    height, width = bb_dim
    if (len(gt)<=3):
        return huber_loss(np.abs(len(gt)-len(pred)))
    
    vor = Voronoi(gt)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    error = []
    for pt, region in zip(gt,regions):
        voronoi_tess = Polygon(vertices[region])
        center_point = Point(pt)
        rectangle = box(center_point.x - width / 2, center_point.y - height / 2,
                    center_point.x + width / 2, center_point.y + height / 2)
        
        intersection = rectangle.intersection(voronoi_tess)

        if not intersection.is_empty:
            intersection_coords = np.array(intersection.exterior.coords)
            poly_path = Path(intersection_coords)
            
            miscount = abs(1-sum([poly_path.contains_point(point) for point in pred]))
            error.append(miscount)
            
    if (len(error)==0):
        return huber_loss(np.abs(len(gt)))
    return huber_loss(np.array(error))


def generate_voronoi_im(img_shape, points):
    
    yy, xx = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
    grid_points = np.column_stack([yy.ravel(), xx.ravel()])

    distances = np.linalg.norm(grid_points[:, None] - points, axis=2)

    nearest_index = np.argmin(distances, axis=1)
    voronoi_image = nearest_index.reshape(img_shape)

    return np.flipud(voronoi_image)

def generate_perp_line(image_shape, point1, point2):
    img = np.zeros(image_shape, dtype=np.uint8)

    midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])

    perp_slope = -1 / (slope + 1e-5)

    x_values = np.linspace(0, image_shape[1] - 1, 100)
    y_values = perp_slope * (x_values - midpoint[0]) + midpoint[1]
    x_values = np.clip(x_values, 0, image_shape[1] - 1)
    y_values = np.clip(y_values, 0, image_shape[0] - 1)
    rr, cc, _ = line_aa(np.round(y_values[0]).astype(int), np.round(x_values[0]).astype(int), np.round(y_values[-1]).astype(int), np.round(x_values[-1]).astype(int))
    img[rr,cc]=1
    return img

def bb_inter_img(gt, img_h, img_w, pseudo_bb, bb_dim=(10,10)):
    try:
        vor = Voronoi(gt)
    except scipy.spatial.qhull.QhullError:
        bb_img = np.zeros((img_h, img_w))
        return bb_img, bb_img
        
    regions, vertices = voronoi_finite_polygons_2d(vor)
    voronoi_array = find_boundaries(generate_voronoi_im((img_h,img_w), gt))
    
    bb_img = np.zeros((img_h, img_w))

    for pt, region in zip(gt, regions):
        voronoi_tess = Polygon(vertices[region])
        center_point = Point(pt)
        if (len(pseudo_bb)==0):
            rectangle = box(center_point.x - bb_dim[1] / 2, center_point.y - bb_dim[0] / 2,
                    center_point.x + bb_dim[1] / 2, center_point.y + bb_dim[0] / 2)
        else:
            rectangle = box(*pseudo_bb[tuple(pt)])
            
        intersection = rectangle.intersection(voronoi_tess)
        if not intersection.is_empty:
            intersection_coords = np.array(intersection.exterior.coords)
            poly_path = Path(intersection_coords)

            x, y = np.indices((img_h, img_w))
            grid_pts = np.column_stack((x.flatten(), y.flatten()))
            pts_inside = poly_path.contains_points(grid_pts)
            img_mask = pts_inside.reshape(img_h, img_w)
            img_mask = img_mask.astype(np.float32)
            bb_img += img_mask
            
    bb_img[bb_img>=2]=0
    
    return bb_img, voronoi_array

def bb_celoss(points, probs, pseudo_bb, bb_dim=(10, 10)):
    blobs = get_blobs(probs, p=0.5)  
    img_h, img_w = probs.shape
    gt = non_zero_pixel_positions(points)
    
    bb_img = np.zeros((img_h, img_w))
    
    if len(gt)==0:
        return []
    elif len(gt)<3:
        for pt in gt:
            center_point = Point(pt)
            if (len(pseudo_bb)==0):
                bb = box(center_point.x - bb_dim[1] / 2, center_point.y - bb_dim[0] / 2,
                        center_point.x + bb_dim[1] / 2, center_point.y + bb_dim[0] / 2)
            else:
                bb = box(*pseudo_bb[tuple(pt)])
                
            bb = np.array(bb.exterior.coords)
            poly_path = Path(bb)
            x, y = np.indices((img_h, img_w))
            grid_pts = np.column_stack((x.flatten(), y.flatten()))
            pts_inside = poly_path.contains_points(grid_pts)
            img_mask = pts_inside.reshape(img_h, img_w)
            img_mask = img_mask.astype(np.float32)
            bb_img += img_mask
            
        if (len(gt)==1):
            voronoid_array = np.zeros((img_h, img_w))
        else:
            point1, point2 = gt[0], gt[1]
            voronoid_array = generate_perp_line((img_h, img_w), point1, point2)
                                       
    else:
        bb_img, voronoid_array = bb_inter_img(gt, img_h, img_w, pseudo_bb)
            
    bb_img_inv = np.logical_not(bb_img)
  
    ceLoss = []
    if (len(gt)>1): 
        ceLoss = [{'scale': len(gt)-1, 'ind_list':np.where(voronoid_array.ravel())[0], 'label':0}]
    
  
    outside_region_pred = np.where(np.multiply(bb_img_inv, blobs))[0]
    if len(outside_region_pred) == 0:
        outside_region_pred = np.where(bb_img_inv)[0]

    if len(outside_region_pred) > 0:
        ceLoss += [{'scale': len(gt)-1, 'ind_list': outside_region_pred, 'label': 0}]
    else:
        ceLoss += []

    inside_region_pred = np.where(np.multiply(bb_img, blobs))[0]
    if len(inside_region_pred) > 0:
        ceLoss.append({'scale': len(gt)-1, 'ind_list': inside_region_pred, 'label': 1})

    return ceLoss
    
