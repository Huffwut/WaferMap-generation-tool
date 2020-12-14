# WaferMap-generation-tool
Work in progress tool. Mainly used to visualize wafermaps when given a nice DataFrame.
If you use it, would be nice to have some credit for this :)

![DataFrame](link)

Main requirements are, to have a data column, e.g. Resistance, and X and Y, which would display on a 2D graph, for that specific die the Resistance value.

![WaferMap](link)

![Wafermap](link)

## Basic usage:

```
import pandas as pd
import wafermap as w

data = pd.read_csv('./some_nice_dataset')

wafer = w.Wafermap(data) # load dataframe into the library

wafer.attempt_image() # generates values for plotting inside the library, (it is being stored)

# default arguments
wafer.plot_wafer(self, c_color='jet', font_size=22, cmap_min=None, cmap_max=None, save=None)

# c_color - choose from matplotlib colormap 
# cmap_min - lowest value within the plot (scaling colormap)
# cmap_max - highest value within the plot
# save - whether to save the plots to a folder, e.g. write save='./plots'

```