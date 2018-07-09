# VanillaGAN-TensorFlow
This repository is Tensorflow implementation of Ian J. Goodfellow's [Generative Adversarial Nets, NIPS2014](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). 

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/42316086-e7223f64-8083-11e8-8653-2e9e52bf3e79.png" width=600)
</p>
  
## Requirements
- tensorflow 1.18.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- pickle 0.7.4

## Applied GAN Structure
1. **Structure for MNIST dataset**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/42433142-f05aada2-8388-11e8-954a-1737e82b6d35.png" width=700>
</p>

2. **Structure for CIFAR10 dataset**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/42433159-02d90ea6-8389-11e8-85fb-d506d177a1cb.png" width=750>
</p>

## Generated Numbers
1. **MNIST**
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/42432259-a0857270-8384-11e8-8d13-bd6a0239c1d9.png" width=800>
</p>

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/42432319-ed0ddd80-8384-11e8-83df-dcb1407cbe65.png" width=800>
</p>

2. **CIFAR10**
- **Note**: The following generated results are very bad. One reason is that we applied a shallow network, 3 fully connected network, and another reason maybe the big image dimension. The generated image size is 32x32x3.

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/42433286-9bacedbe-8389-11e8-9729-59dd9e98158f.png" width=800>
</p>

## Documentation
### Download Dataset
MNIST and CIFAR10 dataset will be downloaded automatically if in a specific folder there are no dataset.

### Directory Hierarchy
``` 
.
├── src
│   ├── cache.py
│   ├── cifar10.py
│   ├── dataset.py
│   ├── dataset_.py
│   ├── download.py
│   ├── main.py
│   ├── solver.py
│   ├── tensorflow_utils.py
│   ├── utils.py
│   └── vanillaGAN.py
```  
**src**: source codes of vanillaGAN

### Training Vanilla GAN
Use `main.py` to train a vanilla GAN network. Example usage:

```
python main.py --is_train true
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `512`
 - `dataset`: dataset name for choice [mnist|cifar10], default: `mnist`
 - `is_train`: 'training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: momentum term of Adam, default: `0.5`
 - `z_dim`: dimension of z vector, default: `100`
 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_size`: sample size for check generated image quality, default: `64`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
### Evaluate Vanilla GAN
Use `main.py` to evaluate a vanilla GAN network. Example usage:

```
python main.py --is_train false --load_model folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018vanillagan,
    author = {Cheng-Bin Jin},
    title = {Vanilla GAN},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/VanillaGAN-TensorFlow}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

### License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
