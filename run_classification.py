import classification
import classification.utilities
import classification.plotting
import classification.generator
import classification.training
import os
import numpy as np

def run_classification(settings):

    # Make directory for classification, if needed, and change wokring directory to it
    orig_dir = os.getcwd()
    os.makedirs(os.path.dirname(settings["classification"]["output_dir_path"]), exist_ok=True)
    os.chdir(settings["classification"]["output_dir_path"])

    # Create symbolic links to all available 3DThesis results.
    # This is to maintain compatibility with the directory
    # structurue that the classification package expects
    os.makedirs("3dthesis", exist_ok=True)
    for result in settings["3DThesis"]["results"]:
        copy_path = os.path.join("3dthesis", os.path.basename(result))
        if not os.path.exists(copy_path): os.symlink(result, copy_path)

    # Setup folder structure
    classification.utilities.folder_setup()

    # Generate voxel training data from 3DThesis data
    voxelTrainingData = classification.generator.make_voxel_training_data(plot=True)

    # Train bnpy voxel classification model
    voxelData, voxelModelPath, nClusterV = classification.training.train_voxel_classifier(
                                                voxelTrainingData, 
                                                dpi=300, 
                                                loadModel=True, 
                                                plot=True, 
                                                sF=0.5, 
                                                gamma=0.8,
                                                modelInitDir="randexamplesbydist")

    # Generate supervoxel training data from the voxel classification data
    supervoxelTrainingData = classification.generator.make_supervoxel_training_data(
        voxelModelPath, 
        voxelStep=0.0125, 
        supervoxelStep=0.25, 
        dpi=300, 
        plot=True)

    # Train supervoxel classification model and generate plots of the classification results
    supervoxelDatasets, _, nClusterSV = classification.training.train_supervoxel_classifier(
                                                                    supervoxelTrainingData, 
                                                                    loadModel=False, 
                                                                    dpi=300, 
                                                                    plot=True, 
                                                                    sF=0.5, 
                                                                    gamma=0.8,
                                                                    modelInitDir="randexamplesbydist")

    # Run post-processing plotting scripts
    if True:
        nrows = int(np.floor(np.sqrt(nClusterV)))
        ncols = int(np.ceil(nClusterV/nrows))
        for id in range(len(supervoxelDatasets)):
            classification.plotting.combined_composition_colormesh(id, nrows=nrows, ncols=ncols, dpi=150)

    status = "Complete"
    
    # Return to original working directory
    os.chdir(orig_dir)
    return status