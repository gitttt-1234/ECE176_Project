<h1 align="center">
  <b>Latent Constraints for Data Generation</b><br>
</h1>


### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 ([GitHub Repo](https://github.com/PyTorchLightning/pytorch-lightning/tree/deb1581e26b7547baf876b7a94361e60bb200d32))
- CUDA enabled computing device


### Usage
```

$ python run.py -c configs/<config-file-name.yaml>
```

run.py - train vae model on CIFAR10
<br>run_vaemnist.py - train vae model on MNIST
<br>run_ac.py - train actor critic model
<br>run_ac_mnist.py - train actor critic model for MNIST
<br>train_classifier - train the classifier on CIFAR 10
<br>train_classifier_mnist - train the classifier on MNIST




