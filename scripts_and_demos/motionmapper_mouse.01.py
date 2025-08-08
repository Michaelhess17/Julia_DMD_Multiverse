#!/usr/bin/env python3
"""
Mouse Movement Analysis using MotionMapper

This script processes mouse pose tracking data from DLC h5 files, performs dimensionality
reduction, and embeds the data into a low-dimensional space for behavioral analysis.

The processing pipeline includes:
1. Loading and preprocessing tracking data
2. Applying median filtering and spline smoothing
3. Ego-centering coordinates relative to body position
4. PCA dimensionality reduction
5. Wavelet transformation
6. Embedding using t-SNE or UMAP
7. Watershed segmentation for behavioral mapping

Usage:
    python motionmapper_mouse.py

Author: Michael
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
import motionmapperpy as mmpy
import hdf5storage
import tqdm.auto as tqdm
from moviepy.editor import VideoFileClip
from matplotlib.animation import FuncAnimation

# Set project paths
PROJECT_PATH = "/home/michael/Synology/Julia/Julia_DMD_Multiverse/scripts_and_demos/data_cleaning/mouse_data"
H5_DIR = "/home/michael/Synology/Python/Gait-Signatures/data/Mouse_Data/"
OUTPUT_DIR = os.path.join(PROJECT_PATH, "Projections")

# Create necessary directories
mmpy.createProjectDirectory(PROJECT_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing parameters
FILTER_SIZE = 21       # Median filter size
SPLINE_SMOOTH = 5000   # Spline smoothing factor
CENTER_SMOOTH = 30000  # Smoothing factor for body center tracking
N_PCA_COMPONENTS = 5   # Number of PCA components to keep

def load_h5_file(h5_path):
    """
    Load and preprocess an h5 file with pose tracking data.
    
    Args:
        h5_path: Path to h5 file
        
    Returns:
        DataFrame with pose data, or None if file doesn't have enough frames
    """
    h5_df = pd.read_hdf(h5_path).iloc[-16500:-500]  # Skip first and last sections
    if h5_df.shape[0] < 16000:
        print(f"Skipping {h5_path}: not enough frames ({h5_df.shape[0]})")
        return None
    return h5_df

def extract_xy_coordinates(h5_df):
    """
    Extract x,y coordinates for each body part from a DataFrame.
    
    Args:
        h5_df: DataFrame with pose data
        
    Returns:
        xy_arr: Array of shape (n_frames, n_bodyparts, 2) with x,y coordinates
        bodyparts: List of bodypart names
    """
    # Extract unique bodyparts from the DataFrame columns
    bodyparts = h5_df.columns.get_level_values('bodyparts').unique().tolist()
    
    # Extract x and y coordinates for each bodypart and stack into array
    xy_arr = np.stack([
        h5_df.xs('x', axis=1, level='coords').loc[:, (slice(None), bodyparts)].values,
        h5_df.xs('y', axis=1, level='coords').loc[:, (slice(None), bodyparts)].values
    ], axis=-1)
    
    # Reorder to match bodyparts list
    xy_arr = xy_arr[:, [h5_df.xs('x', axis=1, level='coords').columns.get_level_values('bodyparts').tolist().index(bp) 
                        for bp in bodyparts], :]
    return xy_arr, bodyparts

def assign_bodypart_indices(bodyparts):
    """
    Group body parts into categories (nose, ear, tail, paw).
    
    Args:
        bodyparts: List of bodypart names
        
    Returns:
        Tuple of indices for different body part categories
    """
    nose_inds = [i for i, bp in enumerate(bodyparts) if bp.startswith('nose')]
    ear_inds = [i for i, bp in enumerate(bodyparts) if bp.startswith('ear')]
    tail_inds = [i for i, bp in enumerate(bodyparts) if bp.startswith('tail')]
    paw_inds = [i for i, bp in enumerate(bodyparts) 
               if i not in nose_inds + ear_inds + tail_inds and 'bodycenter' not in bp]
    return nose_inds, ear_inds, tail_inds, paw_inds

def filter_xy_coordinates(xy_arr, filter_size):
    """
    Apply median filter to position data.
    
    Args:
        xy_arr: Array of shape (n_frames, n_bodyparts, 2) with x,y coordinates
        filter_size: Size of median filter kernel
        
    Returns:
        Filtered array of same shape as input
    """
    xy_filtered = np.zeros_like(xy_arr)
    for i in range(xy_arr.shape[1]):
        xy_filtered[:, i, 0] = median_filter(xy_arr[:, i, 0], size=filter_size)
        xy_filtered[:, i, 1] = median_filter(xy_arr[:, i, 1], size=filter_size)
    return xy_filtered

def smooth_xy_coordinates(xy_arr, s_factor, downsample_factor=3):
    """
    Apply spline smoothing to position data.
    
    Args:
        xy_arr: Array with x,y coordinates
        s_factor: Spline smoothing factor
        downsample_factor: Factor to downsample data before spline fitting
        
    Returns:
        Smoothed array of same shape as input
    """
    xy_smooth = np.zeros_like(xy_arr)
    for i in range(xy_arr.shape[1]):
        x_spline = UnivariateSpline(
            np.arange(xy_arr.shape[0])[::downsample_factor], 
            xy_arr[::downsample_factor, i, 0], 
            s=s_factor
        )
        y_spline = UnivariateSpline(
            np.arange(xy_arr.shape[0])[::downsample_factor], 
            xy_arr[::downsample_factor, i, 1], 
            s=s_factor
        )
        xy_smooth[:, i, 0] = x_spline(np.arange(xy_arr.shape[0]))
        xy_smooth[:, i, 1] = y_spline(np.arange(xy_arr.shape[0]))
    return xy_smooth

def get_body_centers(xy_arr, paw_inds, bodyparts):
    """
    Compute body centers using the median position of paws.
    
    Args:
        xy_arr: Array with x,y coordinates
        paw_inds: Indices of paws
        bodyparts: List of bodypart names
        
    Returns:
        Tuple of arrays for side and bottom view body centers
    """
    bodycenter_side = np.median(
        xy_arr[:, [i for i in paw_inds if 'side' in bodyparts[i]], :], 
        axis=1
    )
    bodycenter_bottom = np.median(
        xy_arr[:, [i for i in paw_inds if 'bottom' in bodyparts[i]], :], 
        axis=1
    )
    return bodycenter_side, bodycenter_bottom

def ego_center_coordinates(xy_arr, bodycenter_side, bodycenter_bottom, bodyparts, use_inds):
    """
    Create ego-centered coordinates by subtracting body center position.
    
    Args:
        xy_arr: Array with x,y coordinates
        bodycenter_side: Body center coordinates (side view)
        bodycenter_bottom: Body center coordinates (bottom view)
        bodyparts: List of bodypart names
        use_inds: Indices of body parts to use
        
    Returns:
        Ego-centered coordinates
    """
    xy_ego = xy_arr[:, use_inds, :].copy()
    for i, bp in enumerate([bodyparts[ind] for ind in use_inds]):
        if 'side' in bp:
            xy_ego[:, i, :] -= bodycenter_side
        elif 'bottom' in bp:
            xy_ego[:, i, :] -= bodycenter_bottom
        else:
            print(f"Warning: Body part {bp} not recognized for ego-centering")
    return xy_ego

def compute_angles(xy_ego, bodyparts, use_inds):
    """
    Compute joint angles between elbow, wrist, and paw.
    
    Args:
        xy_ego: Ego-centered coordinates
        bodyparts: List of bodypart names
        use_inds: Indices of body parts to use
        
    Returns:
        Array of joint angles
    """
    # Identify joint indices
    wrist_inds = [i for i in use_inds if "wrist" in bodyparts[i]]
    elbow_inds = [i for i in use_inds if "elbow" in bodyparts[i]]
    paw_inds = [i for i in use_inds 
                if ("Lside" in bodyparts[i] or "Rside" in bodyparts[i]) 
                and i not in (wrist_inds + elbow_inds)]
    
    assert len(wrist_inds) == len(elbow_inds) == len(paw_inds) == 4, \
        "Mismatch in wrist, elbow, and paw counts. Should be 4 each."
    
    # Map indices
    ind_map_all_to_use = {i: j for j, i in enumerate(use_inds)}
    
    # Initialize angles array
    angles = np.zeros((xy_ego.shape[0], 4))  # 4 limbs
    
    # Calculate angles
    for i, (elbow, wrist, paw) in enumerate(zip(elbow_inds, wrist_inds, paw_inds)):
        # Calculate vectors
        vec1 = xy_ego[:, ind_map_all_to_use[wrist], :] - xy_ego[:, ind_map_all_to_use[elbow], :]
        vec2 = xy_ego[:, ind_map_all_to_use[paw], :] - xy_ego[:, ind_map_all_to_use[wrist], :]
        
        # Calculate dot product and norms
        dot_product = np.sum(vec1 * vec2, axis=1)
        norm1 = np.linalg.norm(vec1, axis=1)
        norm2 = np.linalg.norm(vec2, axis=1)
        
        # Calculate angle
        angles[:, i] = np.arccos(dot_product / (norm1 * norm2))
        
    return angles

def makeGroupsAndSegments(watershedRegions, zValLens, min_length=10, max_length=100):
    """
    Create groups and segments from watershed regions.
    
    Args:
        watershedRegions: Array of region assignments
        zValLens: Lengths of each file segment
        min_length: Minimum segment length
        max_length: Maximum segment length
        
    Returns:
        Array of segment groups
    """
    inds = np.zeros_like(watershedRegions)
    start = 0
    for l in zValLens:
        inds[start:start + l] = np.arange(l)
        start += l
    vinds = np.digitize(np.arange(watershedRegions.shape[0]), 
                      bins=np.concatenate([[0], np.cumsum(zValLens)]), 
                      right=True)

    # Split into segments where region changes
    splitinds = np.where(np.diff(watershedRegions, axis=0) != 0)[0] + 1
    
    # Filter segments by length
    inds = [i for i in np.split(inds, splitinds) 
            if len(i) > min_length and len(i) < max_length]
    wregs = [i[0] for i in np.split(watershedRegions, splitinds) 
             if len(i) > min_length and len(i) < max_length]
    vinds = [i for i in np.split(vinds, splitinds) 
             if len(i) > min_length and len(i) < max_length]
    
    # Group segments by region
    groups = [np.empty((0, 3), dtype=int)] * watershedRegions.max()
    for wreg, tind, vind in zip(wregs, inds, vinds):
        if np.all(vind == vind[0]):
            groups[wreg - 1] = np.concatenate(
                [groups[wreg - 1], np.array([vind[0], tind[0] + 1, tind[-1] + 1])[None, :]])
    
    groups = np.array([[g] for g in groups], dtype=object)
    return groups

def process_h5_files(downsample_factor=3):
    """
    Process all h5 files to extract ego-centered pose data.
    
    Returns:
        Tuple containing:
        - List of ego-centered data arrays
        - List of bodypart names for each file
        - List of valid h5 files processed
    """
    h5_files = sorted(glob.glob(os.path.join(H5_DIR, "**/*.h5"), recursive=True))
    print(f"Found {len(h5_files)} h5 files")
    
    all_xy_ego = []
    file_bodyparts = []
    file_lens = []
    valid_h5_files = []
    
    print("Processing files with robust body center smoothing...")
    
    for h5_path in tqdm.tqdm(h5_files):
        # Load h5 file
        h5_df = load_h5_file(h5_path)
        if h5_df is None:
            continue
        
        # Extract coordinates
        xy_arr, bodyparts = extract_xy_coordinates(h5_df)
        
        # Get body part indices
        nose_inds, ear_inds, tail_inds, paw_inds = assign_bodypart_indices(bodyparts)
        use_inds = [i for i in paw_inds + tail_inds[:5]]  # Use paws and first 5 tail segments
        
        # Apply median filter
        xy_arr[:, use_inds, 0] = median_filter(xy_arr[:, use_inds, 0], size=(FILTER_SIZE, 1))
        xy_arr[:, use_inds, 1] = median_filter(xy_arr[:, use_inds, 1], size=(FILTER_SIZE, 1))
        xy_arr = xy_arr[:, use_inds, :]  # Keep only used body parts

        # Apply spline smoothing
        for dim in range(2):  # 0 for x, 1 for y
            for i in range(xy_arr.shape[1]):
                xy_arr[:, i, dim] = UnivariateSpline(
                    np.arange(xy_arr.shape[0])[::downsample_factor], 
                    xy_arr[::downsample_factor, i, dim], 
                    s=SPLINE_SMOOTH
                )(np.arange(xy_arr.shape[0]))
        
        # Robust body center estimation
        paw_data_side_x = xy_arr[:, [i for i in range(len(use_inds)) if 'side' in bodyparts[use_inds[i]]], 0]
        paw_data_side_y = xy_arr[:, [i for i in range(len(use_inds)) if 'side' in bodyparts[use_inds[i]]], 1]
        
        paw_data_bottom_x = xy_arr[:, [i for i in range(len(use_inds)) if 'bottom' in bodyparts[use_inds[i]]], 0]
        paw_data_bottom_y = xy_arr[:, [i for i in range(len(use_inds)) if 'bottom' in bodyparts[use_inds[i]]], 1]

        time = np.arange(paw_data_side_x.shape[0])
        time_ds = time[::downsample_factor]

        # Median across paws for each frame
        body_center_side_x_robust = np.median(paw_data_side_x, axis=1)
        body_center_side_y_robust = np.median(paw_data_side_y, axis=1)

        body_center_bottom_x_robust = np.median(paw_data_bottom_x, axis=1)
        body_center_bottom_y_robust = np.median(paw_data_bottom_y, axis=1)

        # Spline smoothing for body center
        spl_side_x = UnivariateSpline(time_ds, body_center_side_x_robust[::downsample_factor], s=CENTER_SMOOTH)
        spl_side_y = UnivariateSpline(time_ds, body_center_side_y_robust[::downsample_factor], s=CENTER_SMOOTH)
        smoothed_body_center_side_x = spl_side_x(time)
        smoothed_body_center_side_y = spl_side_y(time)

        spl_bottom_x = UnivariateSpline(time_ds, body_center_bottom_x_robust[::downsample_factor], s=CENTER_SMOOTH)
        spl_bottom_y = UnivariateSpline(time_ds, body_center_bottom_y_robust[::downsample_factor], s=CENTER_SMOOTH)
        smoothed_body_center_bottom_x = spl_bottom_x(time)
        smoothed_body_center_bottom_y = spl_bottom_y(time)

        # Subtract robust body center from all coordinates
        xy_ego = xy_arr.copy()
        for dim in range(xy_ego.shape[1]):
            if 'side' in bodyparts[use_inds[dim]]:
                xy_ego[:, dim, 0] -= smoothed_body_center_side_x
                xy_ego[:, dim, 1] -= smoothed_body_center_side_y
            elif 'bottom' in bodyparts[use_inds[dim]]:
                xy_ego[:, dim, 0] -= smoothed_body_center_bottom_x
                xy_ego[:, dim, 1] -= smoothed_body_center_bottom_y
            else:
                print(f"Warning: Body part {bodyparts[use_inds[dim]]} not recognized for ego-centering")
        
        # Save results
        all_xy_ego.append(xy_ego.reshape(xy_ego.shape[0], -1))
        file_bodyparts.append([bodyparts[i] for i in use_inds])
        file_lens.append(xy_ego.shape[0])
        valid_h5_files.append(h5_path)
    
    print(f"Processed {len(valid_h5_files)} valid files out of {len(h5_files)}")
    return all_xy_ego, file_bodyparts, valid_h5_files

def perform_pca(all_xy_ego):
    """
    Perform PCA on ego-centered data.
    
    Args:
        all_xy_ego: List of ego-centered data arrays
        
    Returns:
        PCA projections of the data
    """
    # Concatenate all data for PCA
    all_xy_ego_concat = np.concatenate(all_xy_ego, axis=0)
    print(f"Total frames for PCA: {all_xy_ego_concat.shape[0]}")
    
    # Center and scale data
    all_xy_ego_concat -= all_xy_ego_concat.mean(axis=0)
    all_xy_ego_concat /= all_xy_ego_concat.std(axis=0)
    
    # Fit PCA
    print("Fitting PCA...")
    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(all_xy_ego_concat)
    
    # Transform data
    y_all = np.stack([pca.transform(data)[:, :N_PCA_COMPONENTS] for data in all_xy_ego])
    
    # Report explained variance
    print(f"Explained variance (first {N_PCA_COMPONENTS}):", 
          np.cumsum(pca.explained_variance_ratio_)[:N_PCA_COMPONENTS])
    
    return y_all

def configure_motionmapper_parameters():
    """
    Configure MotionMapper parameters.
    
    Returns:
        MotionMapper parameters object
    """
    parameters = mmpy.setRunParameters()
    
    # Basic parameters
    parameters.projectPath = PROJECT_PATH
    parameters.method = 'UMAP'  # Can be 'TSNE' or 'UMAP'
    parameters.minF = 0.5       # Minimum frequency for Morlet Wavelet Transform
    parameters.maxF = 100       # Maximum frequency for wavelet transform
    parameters.samplingFreq = 330  # Sampling frequency (FPS) of data
    parameters.numPeriods = 50  # Number of dyadically spaced frequencies
    parameters.pcaModes = N_PCA_COMPONENTS  # Number of PCA components
    parameters.numProcessors = -1  # Use all available cores
    parameters.useGPU = 0  # No GPU
    parameters.training_numPoints = 5000  # Points in mini-trainings
    
    # Memory parameters
    parameters.trainingSetSize = 50000  # Training set size
    parameters.embedding_batchSize = 45000  # Batch size for re-embedding
    
    # t-SNE parameters
    parameters.tSNE_method = 'barnes_hut'
    parameters.perplexity = 512
    parameters.maxNeighbors = 200
    parameters.kdNeighbors = 80
    parameters.training_perplexity = 256
    
    # UMAP parameters
    parameters.n_neighbors = 64
    parameters.train_negative_sample_rate = 30
    parameters.embed_negative_sample_rate = 30
    parameters.min_dist = 0.05
    
    return parameters

def save_projections(y_all, valid_h5_files):
    """
    Save PCA projections to disk.
    
    Args:
        y_all: PCA projections
        valid_h5_files: List of valid h5 files
    """
    for i, h5_path in enumerate(valid_h5_files):
        print(f"Saving projections for file {i+1}/{len(valid_h5_files)}")
        relative_path = os.path.relpath(h5_path, H5_DIR)
        unique_base = os.path.splitext(relative_path)[0].replace(os.path.sep, '_')
        savepath = os.path.join(OUTPUT_DIR, f"{unique_base}_pcaModes.mat")
        hdf5storage.savemat(savepath, {"projections": y_all[i]})

def run_motionmapper_pipeline(parameters, y_all):
    """
    Run the MotionMapper pipeline on PCA projections.
    
    Args:
        parameters: MotionMapper parameters
        y_all: PCA projections
    """
    # Calculate wavelets for the first file
    print("Calculating wavelets...")
    wlets, freqs = mmpy.findWavelets(
        y_all[0], 
        y_all[0].shape[1], 
        parameters.omega0, 
        parameters.numPeriods, 
        parameters.samplingFreq, 
        parameters.maxF, 
        parameters.minF, 
        parameters.numProcessors, 
        parameters.useGPU
    )
    
    # Perform subsampled t-SNE or UMAP
    print(f"Performing {parameters.method}...")
    mmpy.subsampled_tsne_from_projections(parameters, PROJECT_PATH)
    
    # Find embeddings for all files
    print("Finding embeddings...")
    tfolder = parameters.projectPath + '/%s/' % parameters.method
    
    # Load training data
    import h5py
    with h5py.File(tfolder + 'training_data.mat', 'r') as hfile:
        trainingSetData = hfile['trainingSetData'][:].T
    
    # Load training embedding
    with h5py.File(tfolder + 'training_embedding.mat', 'r') as hfile:
        trainingEmbedding = hfile['trainingEmbedding'][:].T
    
    # Set embedding value string based on method
    if parameters.method == 'TSNE':
        zValstr = 'zVals'
    else:
        zValstr = 'uVals'
    
    # Process each file
    projectionFiles = glob.glob(os.path.join(parameters.projectPath, 'Projections', '*pcaModes.mat'))
    for i, proj_file in enumerate(projectionFiles):
        print(f'Finding embeddings for file {i+1}/{len(projectionFiles)}')
        
        # Skip if already processed
        if os.path.exists(proj_file[:-4] + f'_{zValstr}.mat'):
            print('Already processed. Skipping.')
            continue
        
        # Load projections
        projections = hdf5storage.loadmat(proj_file)['projections']
        
        # Find embeddings
        zValues, outputStatistics = mmpy.findEmbeddings(
            projections, 
            trainingSetData, 
            trainingEmbedding, 
            parameters
        )
        
        # Save embeddings
        hdf5storage.write(
            data={'zValues': zValues},
            path='/',
            truncate_existing=True,
            filename=proj_file[:-4] + f'_{zValstr}.mat',
            store_python_metadata=False,
            matlab_compatible=True
        )
        
        # Save statistics
        with open(proj_file[:-4] + f'_{zValstr}_outputStatistics.pkl', 'wb') as hfile:
            pickle.dump(outputStatistics, hfile)
    
    # Find watershed regions
    print("Finding watershed regions...")
    mmpy.findWatershedRegions(
        parameters, 
        minimum_regions=30, 
        startsigma=1.0, 
        pThreshold=[0.33, 0.67],
        saveplot=True, 
        endident='*_pcaModes.mat'
    )

def main():
    """Main function to run the entire pipeline."""
    print("Starting mouse movement analysis...")
    
    # Process h5 files
    all_xy_ego, file_bodyparts, valid_h5_files = process_h5_files()

    # Get body part indices
    nose_inds, ear_inds, tail_inds, paw_inds = assign_bodypart_indices(file_bodyparts[0])
    use_inds = paw_inds + tail_inds[:5]  # Use paws and first 5 tail segments

    # Calculate angles
    angles = np.stack([compute_angles(xy_ego, file_bodypart, use_inds) for (xy_ego, file_bodypart) in zip(all_xy_ego, file_bodyparts)])
    
    # Perform PCA
    y_all = perform_pca(angles)    
    
    # Save projections
    save_projections(y_all, valid_h5_files)
    
    # Configure MotionMapper
    parameters = configure_motionmapper_parameters()
    
    # Run MotionMapper pipeline
    run_motionmapper_pipeline(parameters, y_all)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()