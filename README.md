# ATLASpy: A python library for visualizing volumetric brain data

## Installation

To install:

First clone this repository.

```
git clone https://github.com/allucas/atlaspy.git
```

Then access repository folder on the terminal and type:

```
cd atlaspy
python3 setup.py sdist bdist_wheel
pip3 install dist/atlaspy-0.0.7.tar.gz
```

You should be able to use atlaspy after that.

## Examples

### Plotting the DKT Atlas

```
import atlaspy.core as apy
apy.plot_raw_atlas_left_right('dkt', cmap='Set3')
```

### Plotting the DKT Atlas with custom values

For plotting values from a CSV file or a dataframe, the columns of the CSV must include `atlas_index`, which is the number assigned to the ROI of each atlas, and `roi_value`, which is the values that we would like to plot at that brain region. The atlases used are included in `source_data/atlases/niftis` and their region assignments in `source_data/atlases/luts`. An example input CSV is included in `examples/cortical_thickness_dkt.csv`. To visualize this example use the code below:

```
import atlaspy.core as apy
import pandas as pd
df_values = pd.load_csv('examples/cortical_thickness_dkt.csv')
apy.plot_rois_atlas_lrt('dkt', df_values, cmap='Set3')
```



