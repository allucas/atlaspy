### Creating a custom atlas for ATLASpy

#### Creating STLs 

In order to create the STLs of the atlas I use ITK-SNAP. There are plenty of other ways of doing this, but ITK-SNAP just works really well and it requires only a couple of operations.

1. Open ITK-SNAP![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625165450.png]]
2. Drag and drop the NIFTI file with the segmentation masks (each ROI must have a different label) into ITK-SNAP![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625165604.png]]
3. After the file loads, grab the same file and drag and drop it again into ITK-SNAP. This time press "Load as Segmentation"![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625165801.png]]
4. This will load the data as a segmentation. You should see one color per ROI. To see a 3D rendering of the segmentation, click update in the bottom of the screen![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625165932.png]]
5. Go to Segmentation -> Export as Surface Mesh![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625170012.png]]
6. Select "Export meshes for all labels as separate files" and press "Next>"![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625170039.png]]
7. Provide a name and select STL Mesh File. Make sure this name has no spaces or characters other than letters and numbers. This is needed for ATLASpy to be able to load the data. Select browse to choose the folder where to save the generated STL files. ![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625170205.png]]![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625170508.png]]
8. Press "Finish" and wait for the segmentations to generate.![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625170556.png]]
9. The output folder will contain all of the STL files. They will be named `atlas_nameXXXXX.stl`, where `atlas_name` is the name provided in step 7, and `XXXXX` is the ROI/label value zero padded up to 5 digits. In this example, region with value of 1 was saved as an stl called: `ashsthomas00001.stl`, and region with value of 101 was saved as an stl called `ashsthomas00101.stl`, and so on.![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625170847.png]]

### Loading the atlas into ATLASpy

With the STL files generated, we can then add the atlas to ATLASpy, this is quite easy to do. We use the `add_custom_atlas()` function. The name of the atlas must be the same as the prefix of the STL file, in this case, from above, `ashsthomas`

```python
import atlaspy.core as apy
apy.add_custom_atlas('ashsthomas',
'../atlases/stls/ashsthomas_stls')
```

This will permanently add the atlas `ashsthomas` to ATLASpy, and it can be used freely with any of the other functions provided in ATLASpy

```python
atlas_name = 'ashsthomas'

cmap = 'jet'

camera_position = 'right'

apy.plot_raw_atlas(atlas_name, cmap=cmap, camera_position=camera_position, interactive=True)
```

![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625172754.png]]

We can plot specific regions of our custom atlas by using the atlas index value within a dataframe, making use of the `exclude`  we can exclude certain regions. For example, in this custom atlas, index values 1-5 represent the left hippocampus and 101-105 represent the right hippocampus. Let's assume we have volumetric measurements of the hippocampus, and we have hippocampal atrophy on the right. To plot only these regions with specific values (say volume values for each of the hippocampal subfield), we can do:

```python
import pandas as pd

atlas_indices = [1,2,3,4,5,101,102,103,104,105]

values = [100,134,123,123,123,49,43,23,49,19]

df_values = pd.DataFrame()

df_values['atlas_index'] = atlas_indices

df_values['roi_value'] = values

  

apy.plot_rois_atlas('ashsthomas',df_values, cmap='RdYlBu' ,interactive=True)
```
![[https://github.com/allucas/atlaspy/blob/main/examples/custom_atlas_figures/Pasted%20image%2020230625174129.png]]