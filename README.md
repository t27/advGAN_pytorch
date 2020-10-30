# Updated for CIFAR-10

------------

TODO:

- Update target model
- Increase LR for generator and decrease LR for disc (currently the discriminator is training too fast)
- Change loss function(WLoss)

------------


# Original Readme

------------

# advGAN_pytorch
a Pytorch implementation of the paper "Generating Adversarial Examples with Adversarial Networks" (advGAN).

## training the target model

```shell
python3 train_target_model.py
```

## training the advGAN

```shell
python3 main.py
```

## testing adversarial examples

```shell
python3 test_adversarial_examples.py
```

## results

**attack success rate** in the MNIST test set: **99%**

**NOTE:** My implementation is a little different from the paper, because I add a clipping trick.
