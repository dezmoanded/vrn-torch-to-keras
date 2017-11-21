# vrn-torch-to-keras
Transfer pre-trained VRN model from torch to Keras

### Source of original model
https://github.com/AaronJackson/vrn
Download: http://cs.nott.ac.uk/~psxasj/download.php?file=vrn-unguided.t7

### Script used to parse t7 file
https://github.com/bshillingford/python-torchfile

### Resulting Keras model
https://drive.google.com/file/d/1oh8Zpe4wh00iXcm8ztRsi5ZL6GMkHdjj/view?usp=sharing

### Usage
```python
from keras.models import load_model
import custom_layers
custom_objects = {
    'Conv': custom_layers.Conv,
    'BatchNorm': custom_layers.BatchNorm,
    'UpSamplingBilinear': custom_layers.UpSamplingBilinear
}
model = load_model('vrn-unguided-keras.h5', custom_objects=custom_objects)
```
Input is 3 x 192 x 192 (channels first)<br>
You will need to install h5py.

See [Example-Usage.ipynb](./Example-Usage.ipynb) for a full example using pyplot and visvis

![visvis render](screen_shot.png)
