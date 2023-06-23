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

# dictionary of subcortical regions per atlas
subcortical_dict = {'ho':[8058,2010,2049,3011,3050,4012,4051,5013,5052,6017,6053,6317,6357,6367,6394,7018,7054,7358,7395,8026],
                    'dkt':[17,18,10,11,12,13,49,50,51,52,53,54],
                    'aal':[4101,4102,4201,4202,7001,7002,7011,7012,7021,7022,7101,7102]}


def load_atlas_stls(atlas_name):

    # this function returns the path to the stls for a given atlas
    # Get the absolute path to the current file
    current_file = os.path.abspath(__file__)

    # Determine the base directory of the package
    base_dir = os.path.dirname(current_file)

    stl_dir = os.path.join(base_dir, '../source_data','atlases','stls', atlas_name + '_stls')

    stl_files = os.listdir(stl_dir) 

    stl_files = list(map(lambda x: os.path.join(stl_dir,x), stl_files))

    return stl_files


def get_camera_positions(pos):

    # this function returns the camera position

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

    filename = stl_filename
    pattern = r''+atlas_name+'(\d+)\.stl'
    matches = re.search(pattern, filename)

    if matches:
        digits = matches.group(1)
        digits = int(digits.lstrip('0'))
        return(digits)   

def get_stl_from_index(atlas_name, idx):
    number = str(idx)
    digits = number.zfill(5)
    filename = f'{atlas_name}{digits}.stl'
    return filename


def create_3d_model_from_stls(atlas_name, df_values):
    '''
    df_values: dataframe with two columns - atlas_index, roi_value, exclude
    '''

    atlas_indices = df_values['atlas_index'].values
    roi_values = df_values['roi_value'].values

    if 'exclude' in df_values.columns:
        exclude_list = atlas_indices[df_values['exclude'].values]
    else:
        exclude_list = []

    stl_dir = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'source_data','atlases','stls', atlas_name + '_stls')

    stl_files = list(map(lambda x: os.path.join(stl_dir,get_stl_from_index(atlas_name,x)), atlas_indices))

    pv_list = [] # list of pv objects with each brain region

    for idx,file,roi_val in zip(atlas_indices,stl_files,roi_values):

        if idx not in exclude_list:

            #pv_dict[file.split('.')[0]] = pv.read(os.path.join(stl_dir, file))
            roi = pv.read(file)
            roi.point_data['Segmentation'] = roi_val
            pv_list.append(roi)


    a = pv_list[0]

    brain_model = a.merge(pv_list[1:])
    return brain_model

def set_cmap(cmap):
    print('Setting colormap to ', cmap)
    pv.global_theme.cmap = cmap


def plot_brain_atlas_model(brain_model, camera_position, cmap=None, vmin=None, vmax=None, interactive=False):
    
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

def plot_left_right_brain_atlas_model(brain_model, cmap=None, vmin=None, vmax=None, interactive=False, size=(900,300), scale=1):
    
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

    if interactive==True:
        pl.show()
    else:
        pl.show(jupyter_backend='static')


def plot_left_right_subcortex_model(brain_model, cmap=None, vmin=None, vmax=None, interactive=False, size=(900,300), scale=1):
    
    # same as plot_left_right_brain_atlas_model but it has a zoom so the subcortical structures look bigger

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

    if interactive==True:
        pl.show()
    else:
        pl.show(jupyter_backend='static')




def get_dataframe_from_nifti(atlas_name, nifti_path):
    atlas_dir = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'source_data','atlases','niftis', +atlas_name+'.nii.gz')

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


def plot_atlas_from_nifti(atlas_name, nifti_path, camera_position='left', cmap=None, interactive=False):

    # plot the atlas from a nifti source
    df_values = get_dataframe_from_nifti(atlas_name, nifti_path)
    plot_rois_atlas(atlas_name, df_values, camera_position, cmap=cmap, interactive=interactive)

def plot_rois_atlas(atlas_name, df_values, camera_position='left', cmap=None, interactive=False):
    # create the brain model
    brain_model = create_3d_model_from_stls(atlas_name, df_values)

    # plot the brain model
    plot_brain_atlas_model(brain_model, camera_position=camera_position, cmap=cmap, interactive=interactive)



def plot_atlas_from_nifti_lrt(atlas_name, nifti_path, cmap=None, interactive=False):

    # plot the atlas from a nifti source
    df_values = get_dataframe_from_nifti(atlas_name, nifti_path)
    plot_rois_atlas_lrt(atlas_name, df_values, cmap=cmap, interactive=interactive)


def plot_rois_atlas_lrt(atlas_name, df_values, cmap=None, interactive=False):
    # create the brain model
    brain_model = create_3d_model_from_stls(atlas_name, df_values)

    # plot the brain model
    plot_left_right_brain_atlas_model(brain_model, cmap=cmap, interactive=interactive)

def plot_raw_atlas(atlas_name, camera_position='left',cmap=None, interactive=False):

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
    plot_brain_atlas_model(brain_model, camera_position=camera_position, cmap=cmap, interactive=interactive)

def plot_raw_atlas_left_right(atlas_name, camera_position='left',cmap=None, interactive=False, size=(900,300), scale=1):

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
    plot_left_right_brain_atlas_model(brain_model, cmap=cmap, interactive=interactive, size=size, scale=scale)


def plot_subcortical_brain_regions_lrt(atlas_name, df_values, cmap=None, interactive=False):
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

    plot_left_right_subcortex_model(brain_model, cmap=cmap, interactive=interactive)

def plot_raw_atlas_subcortical_lrt(atlas_name,cmap=None, interactive=False):
    
    # this function plots the subcortical structure of a raw atlas, assigning a random color to each label

    stl_files = load_atlas_stls(atlas_name)

    atlas_indices = list(map(lambda x: get_index_from_stls(atlas_name,x), stl_files))

    # create the dataframe
    df_values = pd.DataFrame()
    df_values['atlas_index'] = atlas_indices

    df_values['roi_value'] = np.random.choice(np.arange(len(atlas_indices)), size=len(atlas_indices), replace=False)
    df_values['exclude'] = False

    plot_subcortical_brain_regions_lrt(atlas_name=atlas_name, df_values=df_values, cmap=cmap, interactive=interactive)