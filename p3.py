#!/usr/bin/env python
import os
import shutil
import sys
import argparse

import numpy as np
from scipy.misc import imread, imsave
from scipy.spatial import Delaunay

#########################################
###########    Skeleton    ##############
#########################################

def intermediate_points(pts1, pts2, fraction):
    """Computes the intermediate point set for a given set of correspondence
    points.  fraction represents the relative weight on start vs end frames, a
    number in the interval [0,1]. 0 means only use the first points, 1 means
    only use the second points.

    The intermediate point set is a linear interpolation of the start and end
    point sets."""

    # Compute and return the intermediate point set.
    intermediate_rows = np.zeros(pts1.shape[0])
    intermediate_cols = np.zeros(pts1.shape[1])
    
    pts1_rows = pts1[...,0]
    pts1_cols = pts1[...,1]
    pts2_rows = pts2[...,0]
    pts2_cols = pts2[...,1]

    intermediate_rows = ((1-fraction) * pts1_rows) + (fraction * pts2_rows) 
    intermediate_cols = ((1-fraction) * pts1_cols) + (fraction * pts2_cols)

    return np.dstack((intermediate_rows, intermediate_cols))

    

def blend(img1, img2, fraction):
    """Blend between img1 and img2 based on the given fraction."""
    # Compute and return the blended image.
    return ((1-fraction) * img1) + (fraction * img2)

def barycentric(points, query):
    """Compute the barycentric coordinates for a query point within the given
    triangle points."""

    # Form the matrix A containing the triangle points and 1s in the proper
    # positions.
    A = np.insert(points, points.shape[1], 1, axis=1)
    A = np.transpose(A)

    # Create the vector b containing the query coordinates as well as a 1.
    b = np.append(query,1)


    # Solve the linear system Ax = b for the barycentric coordinates x.
    # HINT: Look at numpy.linalg.solve.
    return np.linalg.solve(A,b)

def bilinear_interp(image, point):
    """Perform bilinearly-interpolated color lookup in a given image at a given
    point."""

    # Determine row and column values of surrounding pixels
    r1 = np.floor(point[0])
    r2 = r1 + 1
    c1 = np.floor(point[1])
    c2 = c1 + 1

    # Store BGR values at these points
    intensity_tl = image[r1,c1]
    intensity_bl = image[r2,c1]
    intensity_tr = image[r1,c2]
    intensity_br = image[r2,c2]

    # Compute weights
    weight_tl = (point[0] - r1) * (point[1] - c1)
    weight_bl = (r2 - point[0]) * (point[1] - c1)
    weight_tr = (point[0] - r1) * (c2 - point[1])
    weight_br = (r2 - point[0]) * (c2 - point[1])

    return ((intensity_tl * weight_tl) +
            (intensity_bl * weight_bl) +
            (intensity_tr * weight_tr) +
            (intensity_br * weight_br))



def warp(source, source_points, dest_triangulation):
    """Warp the source image so that its correspondences match the destination
    triangulation."""
    result = np.zeros_like(source)

    # Fill in the pixels of the result image.

    # NOTE: This can be done much more efficiently in Python using a series of
    # numpy array operations as opposed to a for loop.

    # HINTS for fast version:
    # * Delaunay.find_simplex can take a list / array of points and look
    #   them up all at once.
    # * You can modify your bilinear_interp() and barycentric() functions to
    #   take arrays of points instead of single points. Express the actions in
    #   these functions via array operations.
    # * Look up numpy.mgrid / meshgrid for tips on how to quickly generate an
    #   array containing all of the points in an image of size [R,C].

    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            # Find the triangle index (look at Delaunay.find_simplex) for the
            # current point in the destination triangulation, then use it to
            # get the points of the destination triangle.

            # Get the indices for the points of the destination triangle
            # (Delaunay.simplices) and use them to get the corresponding points
            # for the source triangle.

            # Compute the barycentric coordinates for the current destination
            # point using the destination triangle's points.

            # Compute the sum of the source points weighted by the barycentric
            # coordinates from the destination triangle.

            # Get the resulting color from the source image via bilinear
            # interpolation, and place it in the result image.
            result[r, c] = 
    return result

def morph(img1, img2, pts1, pts2, fraction):
    """Computes the intermediate morph of the given fraciton between img1
    and img2 using their correspondences."""

    # Compute the intermediate points between the points of the first and
    # second triangulations according to the warp fraction.
    intermediate_pts = intermediate_points(pts1, pts2, fraction) 

    # Compute the triangulation for the intermediate points.
    intermediate_triang = Delaunay(intermediate_pts)

    # Warp the first image to the intermediate triangulation.
    warp1 =

    # Warp the second image to the intermediate triangulation.
    warp2 =

    # Blend the two warped images according to the warp fraction.
    result =

    return result

#########################################
#####    Utility Code      ##############
#########################################

def morph_sequence(start_img, end_img, corrs, n_frames):
    """Computes the n_frames long sequence of intermediate images for the warp
    between start_img and end_img, using the given correspondences in corrs."""

    start_pts, end_pts = corrs

    morph_frames = []
    for frame in range(1, n_frames-1):
        print("Computing intermediate frame %d..." % frame)
        progress = frame/(n_frames-1.)

        intermediate_frame = morph(
                start_img, end_img,
                start_pts, end_pts,
                progress)
    
        imsave("frames/%d.png" % frame, intermediate_frame)
        morph_frames.append( intermediate_frame)

    return [start_img] + morph_frames + [end_img]

def read_corrs(fn):
    pts1 = []
    pts2 = []
    try:
        for line in open(fn).readlines():
            x1, y1, x2, y2 = [int(i) for i in line.strip().split()]
            pts1.append((x1, y1))
            pts2.append((x2, y2))
    except:
        return None

    return (np.array(pts1), np.array(pts2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "CS435 Project 3: Morph one image into another.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument("start_fn", metavar="START_IMG", type=str, help="Filename for the start image.")
    parser.add_argument("end_fn", metavar="END_IMG", type=str, help="Filename for the end image.")
    parser.add_argument("corrs_file", metavar="CORRS_FILE", type=str, help="Text file with corresponding points. Four integers per line separated by spaces: X1 Y1 X2 Y2.")
    parser.add_argument("--n_frames", metavar="N_FRAMES", type=int, default=10, required=False, help="Number of intermediate frames to compute in the morph.")
    parser.add_argument("--outfile", metavar="OUTPUT_FILE", type=str, default="morph.mp4", required=False, help="Name of the file to save the morph movie as. Must end in .mp4")

    args = parser.parse_args()

    if not args.outfile.endswith(".mp4"):
        print("Output filename must end with mp4.")
        exit(1)

    try:
        start_img = imread(args.start_fn)
        start_img = np.dstack(3*[start_img]) if start_img.ndim == 2 else start_img[:, :, :3]
    except:
        print("Error reading start image.")
        exit(1)

    try:
        end_img = imread(args.end_fn)
        end_img = np.dstack(3*[end_img]) if end_img.ndim == 2 else end_img[:, :, :3]
    except:
        print("Error reading end image.")
        exit(1)

    if start_img.shape != end_img.shape:
        print("Start and end images must be the same shape.")
        exit(1)

    corrs = read_corrs(args.corrs_file)
    if corrs is None:
        print("Error reading correspondences file.")
        exit(1)

    # Swap X/Y for Numpy Y/X (Row/Col)
    corrs = [c[:, ::-1] for c in corrs]

    # Add in the 4 corner points of the image to each correspondence set (for
    # triangulation purposes).
    nr, nc = start_img.shape[:2]
    nr -= 1; nc -= 1
    corrs[0] = np.vstack([[[0,0],[nr,0],[0,nc],[nr,nc]], corrs[0]])
    corrs[1] = np.vstack([[[0,0],[nr,0],[0,nc],[nr,nc]], corrs[1]])

    start_img = start_img.astype(np.float32) / start_img.max()
    end_img = end_img.astype(np.float32) / end_img.max()

    if not os.path.exists("frames"):
        os.mkdir("frames")

    morph_img_sequence = morph_sequence(start_img, end_img, corrs, args.n_frames)

    try:
        converter = "avconv" if (os.system("which avconv") == 0) else "ffmpeg"
        codec = "h264" if (os.system("%s -codecs | grep EV | grep h264" % converter) == 0) else "mpeg4 -b 1024k"
        os.system("%s -framerate 15 -i frames/%%d.png -c:v %s -r 30 -pix_fmt yuv420p %s" % (converter, codec, args.outfile))
    except:
        print("Error converting frames to movie.")
        exit(1)
