import pyvista as pv
import os
import pkg_resources
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re
import nibabel as nib
import numpy as np
import requests
from nilearn.image import resample_to_img
import requests
import zipfile

# dictionary of subcortical regions per atlas
subcortical_dict = {'ho':[8058,2010,2049,3011,3050,4012,4051,5013,5052,6017,6053,6317,6357,6367,6394,7018,7054,7358,7395,8026],
                    'dkt':[17,18,10,11,12,13,49,50,51,52,53,54],
                    'aal':[4101,4102,4201,4202,7001,7002,7011,7012,7021,7022,7101,7102]}


def check_for_stls():




    def download_files():

        # Get the absolute path to the current file
        current_file = os.path.abspath(__file__)

        # Determine the base directory of the package
        base_dir = os.path.dirname(current_file)

        stl_dir = os.path.join(base_dir, 'source_data','atlases')

        # Define the URL from which to download the files

        # v1: zip_url = 'http://dl.dropboxusercontent.com/scl/fi/s88fszf6t1q6ef4znl6cr/stls.zip?dl=0&rlkey=j93ehij42d3g0rp1hqn9u0f5t'
        zip_url = 'http://dl.dropboxusercontent.com/scl/fi/a7usav2cmyyskdb339pzu/stls_v2.zip?dl=0&rlkey=3zxd59bginwkkvnte1ojrzfk2'

        # Create the target directory if it doesn't exist
        os.makedirs(stl_dir, exist_ok=True)

        # Download the ZIP file
        zip_path = os.path.join(stl_dir, 'stls.zip')
        response = requests.get(zip_url)
        with open(zip_path, 'wb') as file:
            file.write(response.content)

        # Extract the STL files from the ZIP file preserving folder structure
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(stl_dir)

        # Remove the downloaded ZIP file
        os.remove(zip_path)
    
    # Get the absolute path to the current file
    current_file = os.path.abspath(__file__)

    # Determine the base directory of the package
    base_dir = os.path.dirname(current_file)
    stl_dir = os.path.join(base_dir, 'source_data','atlases','stls')

    if os.path.exists(stl_dir)==False:
        print('Downloading and extracting STL files needed for plotting...')
        download_files()

    




def load_atlas_stls(atlas_name):
    """
    Load the STL files for a given atlas.

    Inputs:
    - atlas_name: Name of the atlas (string)

    Output:
    - stl_files: List of paths to the STL files (list of strings)

    Usage:
    - The function assumes that the STL files for the specified atlas are located in a directory named 'atlas_name_stls'.
    - The function returns a list of paths to the STL files within that directory.
    """
    
    # Check for STLs
    check_for_stls()

    # Get the absolute path to the current file
    current_file = os.path.abspath(__file__)

    # Determine the base directory of the package
    base_dir = os.path.dirname(current_file)

    stl_dir = os.path.join(base_dir, 'source_data','atlases','stls', atlas_name + '_stls')

    stl_files = os.listdir(stl_dir) 

    stl_files = list(map(lambda x: os.path.join(stl_dir,x), stl_files))

    return stl_files


def get_camera_positions(pos):

    """
    Get the camera position to be used when visualizing brain regions.

    Inputs:
    - pos: Position specifier ('right', 'left', 'top') (string)

    Output:
    - cpos: Camera position coordinates (tuple)

    Usage:
    - The function returns the camera position coordinates for the specified position.
    - Valid position specifiers are 'right', 'left', and 'top'.
    """

    if pos=='right':
        cpos = [(426.9990536457896, 6.513542175292969, 13.830495834350586),
 (1.1661758422851562, 6.513542175292969, 13.830495834350586),
 (0.0, 0.0, 1.0)]

    elif pos=='left':
        cpos = [(-426.9990536457896, 6.513542175292969, 13.830495834350586),
 (1.1661758422851562, 6.513542175292969, 13.830495834350586),
 (0.0, 0.0, 1.0)]
        
    elif pos=='top':
        cpos = [(0.5, -16.5, 531.4855618961011),
 (0.5, -16.5, 6.187538146972656),
 (0.0, 1.0, 0.0)]
        
    return cpos

def get_index_from_stls(atlas_name,stl_filename):

    """
    Get the index from the STL filename based on the atlas name.

    Inputs:
    - atlas_name: Name of the atlas (string)
    - stl_filename: Filename of the STL file (string)

    Output:
    - index: Extracted index from the filename (int)

    Usage:
    - The function expects the STL filename to follow the pattern '<atlas_name><index>.stl',
      where <atlas_name> is the specified atlas name and <index> is the extracted index.
    - The function returns the extracted index as an integer.
    """


    filename = stl_filename
    pattern = r''+atlas_name+'(\d+)\.stl'
    matches = re.search(pattern, filename)

    if matches:
        digits = matches.group(1)
        digits = int(digits.lstrip('0'))
        return(digits)   

def get_stl_from_index(atlas_name, idx):

    """
    Get the STL filename based on the atlas name and index.

    Inputs:
    - atlas_name: Name of the atlas (string)
    - idx: Index for the STL file (int)

    Output:
    - filename: STL filename based on the atlas name and index (string)

    Usage:
    - The function generates the STL filename using the specified atlas name and index.
    - The index is zero-padded to a length of 5 digits before being appended to the atlas name.
    - The function returns the generated STL filename as a string.
    """


    number = str(idx)
    digits = number.zfill(5)
    filename = f'{atlas_name}{digits}.stl'
    return filename


def create_3d_model_from_stls(atlas_name, df_values):
    
    """
    Create a 3D model from STL files based on the atlas name and DataFrame values.

    Inputs:
    - atlas_name: Name of the atlas (string)
    - df_values: DataFrame with two mandatory columns - 'atlas_index' and 'roi_value',
                 and an optional column - 'exclude' (DataFrame)

    Output:
    - brain_model: 3D model created from STL files (pyvista.PolyData)

    Usage:
    - The function expects the STL files to be present in a directory named 'atlas_name_stls'.
    - The 'df_values' DataFrame should contain the 'atlas_index' column with the corresponding index values.
    - The 'df_values' DataFrame should also contain the 'roi_value' column with the ROI values for each index.
    - An optional 'exclude' column in the DataFrame can be used to exclude specific indices from the model.
    """

    # check for stls
    check_for_stls()


    atlas_indices = df_values['atlas_index'].values
    roi_values = df_values['roi_value'].values

    if 'exclude' in df_values.columns:
        exclude_list = atlas_indices[df_values['exclude'].values]
    else:
        exclude_list = []



    # Get the absolute path to the current file
    current_file = os.path.abspath(__file__)

    # Determine the base directory of the package
    base_dir = os.path.dirname(current_file)

    stl_dir = os.path.join(base_dir, 'source_data','atlases','stls', atlas_name + '_stls')

    stl_files = list(map(lambda x: os.path.join(stl_dir,get_stl_from_index(atlas_name,x)), atlas_indices))

    pv_list = [] # list of pv objects with each brain region

    for idx,file,roi_val in zip(atlas_indices,stl_files,roi_values):

        if idx not in exclude_list:

            #pv_dict[file.split('.')[0]] = pv.read(os.path.join(stl_dir, file))
            if os.path.exists(file):
                roi = pv.read(file)
                roi.point_data['Segmentation'] = roi_val
                pv_list.append(roi)


    a = pv_list[0]

    brain_model = a.merge(pv_list[1:])
    return brain_model

def set_cmap(cmap):

    """
    Set the colormap for visualization.

    Inputs:
    - cmap: Name of the colormap (string)

    Usage:
    - The function sets the colormap for visualization using the specified name.
    - The colormap can be any of the matplotlib colormaps.
    - The function prints a message indicating the colormap that has been set.
    """

    print('Setting colormap to ', cmap)
    pv.global_theme.cmap = cmap


def plot_brain_atlas_model(brain_model, camera_position, cmap=None, vmin=None, vmax=None, interactive=False):
    """
    Plot a brain atlas model.

    Inputs:
    - brain_model: PyVista mesh object representing the brain model
    - camera_position: Position of the camera for visualization ('right', 'left', or 'top')
    - cmap: Name of the colormap (string) (optional)
    - vmin: Minimum value for the scalar range (float) (optional)
    - vmax: Maximum value for the scalar range (float) (optional)
    - interactive: Flag indicating whether to enable interactive mode (bool) (optional)

    Usage:
    - The function plots the provided brain atlas model using PyVista.
    - The brain model should be a PyVista mesh object created by the 'create_3d_model_from_stls' function.
    - The camera_position parameter specifies the desired camera position for visualization: 'right', 'left', or 'top'.
    - The cmap parameter can be used to set the colormap for visualization (optional).
    - The vmin and vmax parameters can be used to set the scalar range for visualization (optional).
    - By default, the function displays the plot in a static mode (non-interactive) in Jupyter notebooks.
    - If interactive is set to True, the plot will be displayed in an interactive mode (optional).
    """

    # plot a brain model created by create_3d_model_from_stls

    if cmap!=None:
        set_cmap(cmap)
    
    p = pv.Plotter()
    p.set_background('white')
    pv.global_theme.font.color = 'black'

    p.add_mesh(brain_model)
    if ((vmin == None) or (vmax == None))==False:
        p.update_scalar_bar_range([vmin, vmax])

    if interactive==False:
        p.show(jupyter_backend='static', cpos=get_camera_positions(camera_position))
    else:
        p.show(cpos=get_camera_positions(camera_position))



def shift_model(brain_model, translation_vector):
    brain_model_shifted = brain_model.copy()
    brain_model_shifted.points += translation_vector
    return brain_model_shifted

def plot_left_right_top_brain_atlas_model(brain_model, cmap=None, vmin=None, vmax=None, interactive=False, size=(900,300), scale=1):
    """
    Plot the brain atlas model from three different camera positions: left, right, and top.

    Inputs:
    - brain_model: PyVista mesh object representing the brain model
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)
    - size: Size of the plot window (default: (900, 300))
    - scale: Scale factor for the plot window (default: 1)

    Usage:
    - The function plots the brain atlas model from three different camera positions: left, right, and top.
    - The brain model should be a PyVista mesh object created by the 'create_3d_model_from_stls' function.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    - size specifies the size of the plot window (optional). Default is (900, 300).
    - scale is a scale factor for the plot window (optional). Default is 1.
    """

    if cmap!=None:
        set_cmap(cmap)


    pl = pv.Plotter(shape=(1, 3),  window_size=size*scale)

    pl.set_background('white')
    pv.global_theme.font.color = 'black'

    pl.subplot(0, 0)
    actor = pl.add_mesh(shift_model(brain_model,[0,25,0]))
    pl.camera_position = get_camera_positions('left')
    pl.camera.view_angle = 27.0

    pl.subplot(0, 1)
    actor = pl.add_mesh(shift_model(brain_model,[0,25,0]))
    pl.camera_position = get_camera_positions('right')
    pl.camera.view_angle = 27.0


    pl.subplot(0, 2)
    actor = pl.add_mesh(brain_model)
    pl.camera_position = get_camera_positions('top')
    pl.camera.view_angle = 23.0

    if ((vmin == None) or (vmax == None))==False:
        pl.update_scalar_bar_range([vmin, vmax])

    if interactive==True:
        pl.show()
    else:
        pl.show(jupyter_backend='static')


def plot_left_right_top_subcortex_model(brain_model, cmap=None, vmin=None, vmax=None, interactive=False, size=(900,300), scale=1):
    
    """
    Plot the subcortex model from three different camera positions: left, right, and top with a zoom.

    Inputs:
    - brain_model: PyVista mesh object representing the subcortex model
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)
    - size: Size of the plot window (default: (900, 300))
    - scale: Scale factor for the plot window (default: 1)

    Usage:
    - The function plots the subcortex model from three different camera positions: left, right, and top, with a zoom.
    - The brain model should be a PyVista mesh object created by the 'create_3d_model_from_stls' function.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    - size specifies the size of the plot window (optional). Default is (900, 300).
    - scale is a scale factor for the plot window (optional). Default is 1.
    """

    if cmap!=None:
        set_cmap(cmap)


    pl = pv.Plotter(shape=(1, 3),  window_size=size*scale)

    pl.set_background('white')
    pv.global_theme.font.color = 'black'

    pl.subplot(0, 0)
    actor = pl.add_mesh(shift_model(brain_model,[0,15,10]))
    pl.camera_position = get_camera_positions('left')
    pl.camera.view_angle = 15.0

    pl.subplot(0, 1)
    actor = pl.add_mesh(shift_model(brain_model,[0,15,10]))
    pl.camera_position = get_camera_positions('right')
    pl.camera.view_angle = 15.0


    pl.subplot(0, 2)
    actor = pl.add_mesh(brain_model)
    pl.camera_position = get_camera_positions('top')
    pl.camera.view_angle = 15.0

    if ((vmin == None) or (vmax == None))==False:
        pl.update_scalar_bar_range([vmin, vmax])

    if interactive==True:
        pl.show()
    else:
        pl.show(jupyter_backend='static')




def get_dataframe_from_nifti(atlas_name, nifti_path):

    """
    Get the values within each atlas ROI and create a DataFrame using the atlas values.

    Inputs:
    - atlas_name: Name of the atlas (string)
    - nifti_path: Path to the NIfTI image file (string)

    Outputs:
    - df_values: DataFrame containing the atlas indices, ROI values, and exclusion flag

    Usage:
    - The function assumes that all values within the atlas ROI are the same.
    - If you would like the mean value of the ROI instead, use the function get_mean_dataframe_from_nifti().
    """
    
    # Get the absolute path to the current file
    current_file = os.path.abspath(__file__)

    # Determine the base directory of the package
    base_dir = os.path.dirname(current_file)

    atlas_dir = os.path.join(base_dir, 'source_data','atlases','niftis', +atlas_name+'.nii.gz')

    nii_data = nib.load(nifti_path).get_fdata()
    nii_atlas = nib.load(atlas_dir).get_fdata()

    atlas_indices = list(set(list(nii_atlas.ravel()))-{0})

    roi_values_for_indices = []
    for idx in atlas_indices:
        roi_values_for_indices.append(nii_data[nii_atlas==idx][0])

    df_values = pd.DataFrame()
    df_values['atlas_index'] = np.array(atlas_indices).astype(int)
    df_values['roi_value'] = roi_values_for_indices
    df_values['exclude'] = False
    print(df_values)
    return df_values


def get_mean_dataframe_from_nifti(atlas_name, nifti_path, reslice=False):

    """
    Get the mean values within each atlas ROI and create a DataFrame using the atlas values.

    Inputs:
    - atlas_name: Name of the atlas (string)
    - nifti_path: Path to the NIfTI image file (string)
    - reslice (optional): Whether to reslice the NIfTI image to match atlas resolution and orientation (boolean)

    Outputs:
    - df_values: DataFrame containing the atlas indices, ROI values, and exclusion flag

    Usage:
    - The function assumes that all values within the atlas ROI are not the same.
    - It assumes that the NIfTI image is registered to the MNI152NLin2009cAsym_res-01_desc-brain_T1w template.
    - If the NIfTI data is in MNI space but not resliced to the same resolution, set reslice=True for reslicing.
    """

    # Get the absolute path to the current file
    current_file = os.path.abspath(__file__)

    # Determine the base directory of the package
    base_dir = os.path.dirname(current_file)

    atlas_dir = os.path.join(base_dir, 'source_data','atlases','niftis', +atlas_name+'.nii.gz')

    if reslice:
        # use ants to reslice nifti_path to the atlas_dir image, store the data as a numpy array
        atlas_image = nib.load(atlas_dir)
        nii_image = nib.load(nifti_path)

        # Resample the nifti image to match the resolution and orientation of the atlas image
        resampled_nifti = resample_to_img(nii_image, atlas_image, interpolation='linear')

        nii_data = resampled_nifti.get_fdata()
    else:
        nii_data = nib.load(nifti_path).get_fdata()

    nii_atlas = nib.load(atlas_dir).get_fdata()

    atlas_indices = list(set(list(nii_atlas.ravel()))-{0})

    roi_values_for_indices = []
    for idx in atlas_indices:
        roi_values_for_indices.append(np.nanmean(nii_data[nii_atlas==idx]))

    df_values = pd.DataFrame()
    df_values['atlas_index'] = np.array(atlas_indices).astype(int)
    df_values['roi_value'] = roi_values_for_indices
    df_values['exclude'] = False
    print(df_values)
    return df_values


def plot_atlas_from_nifti(atlas_name, nifti_path, camera_position='left', cmap=None,  vmin=None, vmax=None, interactive=False):
    """
    Plot the atlas regions from a NIfTI source file.

    Inputs:
    - atlas_name: Name of the atlas
    - nifti_path: Path to the NIfTI file
    - camera_position: Camera position for visualization (default: 'left')
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)

    Usage:
    - The function reads the NIfTI file specified by nifti_path and extracts the atlas regions using get_dataframe_from_nifti.
    - It then calls plot_rois_atlas to plot the regions of interest (ROIs) using the extracted dataframe and other optional parameters.
    - atlas_name is the name of the atlas, and nifti_path is the path to the NIfTI file.
    - camera_position specifies the camera position for visualization (options: 'right', 'left', 'top'; default: 'left').
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    """

    # plot the atlas from a nifti source
    df_values = get_dataframe_from_nifti(atlas_name, nifti_path)
    plot_rois_atlas(atlas_name, df_values, camera_position, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)


def plot_rois_atlas(atlas_name, df_values, camera_position='left', cmap=None, vmin=None, vmax=None, interactive=False):

    """
    Plot regions of interest (ROIs) on an atlas.

    Inputs:
    - atlas_name: Name of the atlas (string)
    - df_values: DataFrame containing ROI values and indices (pandas DataFrame)
    - camera_position: Position of the camera for visualization ('right', 'left', or 'top') (optional)
    - cmap: Name of the colormap (string) (optional)
    - vmin: Minimum value for the scalar range (float) (optional)
    - vmax: Maximum value for the scalar range (float) (optional)
    - interactive: Flag indicating whether to enable interactive mode (bool) (optional)

    Usage:
    - The function creates a 3D brain model based on the provided atlas name and ROI values DataFrame.
    - The atlas_name parameter specifies the name of the atlas.
    - The df_values should be a pandas DataFrame containing two mandatory columns: 'atlas_index' and 'roi_value'.
    - The camera_position parameter specifies the desired camera position for visualization (optional).
    - The cmap parameter can be used to set the colormap for visualization (optional).
    - The vmin and vmax parameters can be used to set the scalar range for visualization (optional).
    - By default, the function displays the plot in a static mode (non-interactive) in Jupyter notebooks.
    - If interactive is set to True, the plot will be displayed in an interactive mode (optional).
    """
    # create the brain model
    brain_model = create_3d_model_from_stls(atlas_name, df_values)

    # plot the brain model
    plot_brain_atlas_model(brain_model, camera_position=camera_position, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)



def plot_atlas_from_nifti_lrt(atlas_name, nifti_path, cmap=None, vmin=None, vmax=None, interactive=False):
    """
    Plot the atlas regions from a NIfTI source file using LRT (Left-Right-Top) camera position.

    Inputs:
    - atlas_name: Name of the atlas
    - nifti_path: Path to the NIfTI file
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)

    Usage:
    - The function reads the NIfTI file specified by nifti_path and extracts the atlas regions using get_dataframe_from_nifti.
    - It then calls plot_rois_atlas_lrt to plot the regions of interest (ROIs) using the extracted dataframe and other optional parameters.
    - atlas_name is the name of the atlas, and nifti_path is the path to the NIfTI file.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    """
    
    # plot the atlas from a nifti source
    df_values = get_dataframe_from_nifti(atlas_name, nifti_path)
    plot_rois_atlas_lrt(atlas_name, df_values, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)


def plot_rois_atlas_lrt(atlas_name, df_values, cmap=None, vmin=None, vmax=None, interactive=False):

    """
    Plot the regions of interest (ROIs) from an atlas using LRT (Left-Right-Top) camera position.

    Inputs:
    - atlas_name: Name of the atlas
    - df_values: Dataframe containing the atlas indices and ROI values
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)

    Usage:
    - The function creates a 3D brain model using create_3d_model_from_stls, which combines the ROIs from the atlas based on the provided dataframe.
    - It then calls plot_left_right_top_brain_atlas_model to visualize the brain model with LRT camera position.
    - atlas_name is the name of the atlas, and df_values is the dataframe containing the atlas indices and ROI values.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    """


    # create the brain model
    brain_model = create_3d_model_from_stls(atlas_name, df_values)

    # plot the brain model
    plot_left_right_top_brain_atlas_model(brain_model, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)



def plot_raw_atlas(atlas_name, camera_position='left',cmap=None, vmin=None, vmax=None, interactive=False):
    """
    Plot the raw atlas with random colors assigned to each label.

    Inputs:
    - atlas_name: Name of the atlas
    - camera_position: Camera position for visualization (default: 'left')
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)

    Usage:
    - The function plots the raw atlas by assigning a random color to each label.
    - The atlas_name should correspond to the available atlas with its STL files.
    - camera_position specifies the camera position for visualization (optional). Default is 'left'.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    """

    # this function plots the raw atlas, assigning a random color to each label

    stl_files = load_atlas_stls(atlas_name)

    atlas_indices = list(map(lambda x: get_index_from_stls(atlas_name,x), stl_files))

    # create the dataframe
    df_values = pd.DataFrame()
    df_values['atlas_index'] = atlas_indices

    df_values['roi_value'] = np.random.choice(np.arange(len(atlas_indices)), size=len(atlas_indices), replace=False)

    # create the 3d brain atlas model
    brain_model = create_3d_model_from_stls(atlas_name,df_values)

    # plot the brain model
    plot_brain_atlas_model(brain_model, camera_position=camera_position, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)

def plot_raw_atlas_left_right_top(atlas_name, cmap=None, vmin=None, vmax=None, interactive=False, size=(900,300), scale=1):
    """
    Plot the raw atlas with random colors assigned to each label from left, right, and top camera positions.

    Inputs:
    - atlas_name: Name of the atlas
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)
    - size: Size of the plot window (default: (900, 300))
    - scale: Scale factor for the plot window (default: 1)

    Usage:
    - The function plots the raw atlas by assigning a random color to each label from left, right, and top camera positions.
    - The atlas_name should correspond to the available atlas with its STL files.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    - size specifies the size of the plot window (optional). Default is (900, 300).
    - scale is a scale factor for the plot window (optional). Default is 1.
    """

    stl_files = load_atlas_stls(atlas_name)

    atlas_indices = list(map(lambda x: get_index_from_stls(atlas_name,x), stl_files))

    # create the dataframe
    df_values = pd.DataFrame()
    df_values['atlas_index'] = atlas_indices

    df_values['roi_value'] = np.random.choice(np.arange(len(atlas_indices)), size=len(atlas_indices), replace=False)

    # create the 3d brain atlas model
    brain_model = create_3d_model_from_stls(atlas_name,df_values)

    # plot the brain model
    plot_left_right_top_brain_atlas_model(brain_model, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive, size=size, scale=scale)


def plot_subcortical_brain_regions_lrt(atlas_name, df_values, cmap=None, vmin=None, vmax=None, interactive=False):
    """
    Plot the subcortical brain regions of an atlas from left, right, and top camera positions.

    Inputs:
    - atlas_name: Name of the atlas
    - df_values: DataFrame containing the atlas indices and corresponding values
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)

    Usage:
    - The function plots the subcortical brain regions of an atlas from left, right, and top camera positions.
    - The atlas_name should correspond to the available atlas with its STL files.
    - df_values is a DataFrame containing the atlas indices and corresponding values.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    """

    sub_regions = subcortical_dict[atlas_name]
    atlas_regions = df_values['atlas_index'].values
    
    exclude_list = np.zeros(len(atlas_regions)).astype(bool)

    df_values_subcortical = df_values.copy()
    print(df_values)
    # exclude the non-subcortical regions
    for i,reg in enumerate(atlas_regions):
        if reg not in sub_regions:
            exclude_list[i] = True
    
    df_values_subcortical['exclude'] = exclude_list
    
    brain_model = create_3d_model_from_stls(atlas_name=atlas_name, df_values=df_values_subcortical)

    plot_left_right_top_subcortex_model(brain_model, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)

def plot_raw_atlas_subcortical_lrt(atlas_name,cmap=None, vmin=None, vmax=None, interactive=False):
    """
    Plot the subcortical structure of a raw atlas with random colors assigned to each label.

    Inputs:
    - atlas_name: Name of the atlas
    - cmap: Colormap for visualization (default: None)
    - vmin: Minimum value for colormap normalization (default: None)
    - vmax: Maximum value for colormap normalization (default: None)
    - interactive: Flag to enable interactive plotting (default: False)

    Usage:
    - The function plots the subcortical structure of a raw atlas by assigning a random color to each label.
    - The atlas_name should correspond to the available atlas with its STL files.
    - cmap is the colormap for visualization (optional). If not provided, the default colormap is used.
    - vmin and vmax are the minimum and maximum values for colormap normalization (optional). If not provided, the default normalization is used.
    - interactive flag determines whether the plot is interactive or static (default: False).
    """
    # this function plots the subcortical structure of a raw atlas, assigning a random color to each label

    stl_files = load_atlas_stls(atlas_name)

    atlas_indices = list(map(lambda x: get_index_from_stls(atlas_name,x), stl_files))

    # create the dataframe
    df_values = pd.DataFrame()
    df_values['atlas_index'] = atlas_indices

    df_values['roi_value'] = np.random.choice(np.arange(len(atlas_indices)), size=len(atlas_indices), replace=False)
    df_values['exclude'] = False

    plot_subcortical_brain_regions_lrt(atlas_name=atlas_name, df_values=df_values, cmap=cmap, vmin=vmin, vmax=vmax, interactive=interactive)