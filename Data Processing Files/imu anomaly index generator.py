import os
import pandas as pd
from matplotlib.widgets import SpanSelector

import matplotlib.pyplot as plt

filename = "P20V2_F5.csv"
channel = "gz2"

directory = os.path.join(os.path.dirname(os.getcwd()), "Unsupervised-Anomaly-Detection\\TFO Data\\IMU")

file_path = os.path.join(directory, filename)

df = pd.read_csv(file_path)
df_indices = pd.DataFrame(False, columns=[channel], index=df.index)

save_directory = os.path.join(os.path.dirname(os.getcwd()), "Unsupervised-Anomaly-Detection\\TFO Data\\IMU\\Anomaly Indices")
save_path = os.path.join(save_directory, filename.replace('.csv', f'_{channel}_anomaly_indices.csv'))

# Variables to store the start and end indices
start_index = None
end_index = None
done_adding = False  # Flag to indicate if the user is done adding True values

def onselect(xmin, xmax):
    # Convert the x-coordinates to integer indices
    start_index = int(xmin)
    end_index = int(xmax)
    df_indices.loc[start_index:end_index, channel] = True

def on_key(event):
    if event.key == 'enter':
        global done_adding
        done_adding = True
        plt.close()

# Create the plot
fig, ax = plt.subplots()
ax.plot(df[channel])
ax.set_xlabel('Index')
ax.set_ylabel(channel)
ax.set_title('Plot of ' + channel)
ax.set_xlim(df.index.min(), df.index.max())

# Create the SpanSelector
span_selector = SpanSelector(ax, onselect, 'horizontal', useblit=True)

# Register the key press event handler
fig.canvas.mpl_connect('key_press_event', on_key)

plt.xlim(df.index.min(), df.index.max())

plt.show()

# Save df_indices if the user is done adding True values
if done_adding:
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    df_indices.to_csv(save_path)