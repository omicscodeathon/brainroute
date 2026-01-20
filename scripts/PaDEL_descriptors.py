import os
import subprocess


# Define commands for PaDEL
# 'java', '-jar', os.path.join(path, 'PaDEL-Descriptor/PaDEL-Descripto.jar'), '-descriptortypes',
#     os.path.join(path, 'PaDEL-Descriptor/descriptors.xml'),
padel_cmd = [
    'java', '-Xms8G', '-Xmx8G', '-jar', 'scripts/PaDEL-Descriptor/PaDEL-Descriptor.jar', 
    '-descriptortypes','scripts/PaDEL-Descriptor/descriptors.xml', 
    '-dir', 'data/clean_smiles.smi', '-file', 'data/clean_PaDEL_descriptors.csv', '-2d', '-fingerprints', 
    '-removesalt', '-detectaromaticity', '-standardizenitro'    
]

# calculate feature

result = subprocess.run(
    padel_cmd, 
    check=True,  # Raise a CalledProcessError if the command returns a non-zero exit code
)
print('All Features Calculated successfully!')
