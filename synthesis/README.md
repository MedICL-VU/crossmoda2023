# Dynamic network for unpaired image 
Our synthesis network is used for 3D unpaired image translation. It can generate controllable output styles by conditioning on a one-hot code.

Train
"""
python train.py -n YourExperimentName -d YourImageFolder 
"""

Test
"""
python test.py -n YourExperimentName -epoch latest --code 0 0 1 -i YourImageFolder/ImagesA
"""
