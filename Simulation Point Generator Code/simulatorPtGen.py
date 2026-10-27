import numpy as np
import os

start_low_density1 = 2.354
start_low_density2 = 2.4456
end_low_density1 = 2.3548
end_low_density2 = 2.456
high_density_step = 0.0003
region1 = np.linspace(start_low_density1, end_low_density1, 5)*1e9
roi = np.arange(end_low_density1, start_low_density2, high_density_step)*1e9
region2 = np.linspace(start_low_density2, end_low_density2, 5)*1e9

np4csv = np.concatenate((region1, roi, region2))

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'frequencyCSV.csv')
np.savetxt(output_path, np4csv, delimiter=',')

print(f"Saved CSV to: {output_path}")