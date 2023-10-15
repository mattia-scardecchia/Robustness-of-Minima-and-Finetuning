# Robustness of minima and finetuning performance

With the rise of foundation models in deep learning, it is common practice in many applications to finetune a giant model pretrained on an enormous corpus of data to the specific task at hand. Motivated by this observation, in this research project we set out to investigate which properties of the final pretraining configurations can lead to strong finetuning performance.

This is joint work with Dario Filatrella, Rocco Giampetruzzi and Rolf Minardi. The file `report.pdf` contains a detailed description of our methods and findings, with some relevant figures included.

## Experimental setting

We considered the problem of pretraining a ResNet on a subset of ImageNet, and finetuning on CIFAR10. This simple scenario made it possible to perform multiple runs and obtain significant results despite our limited computational budget.
We performed the pretraining using a variety of optimization algorithms, trying to bias the search towards different types of local minima in the loss landscape. Then, we measured various properties of the minima, with a focus on robustness to perturbation (adversarial or random) of weights or inputs. Here is an example:

![](https://github.com/MattiaSC01/Robustness-of-Minima-and-Finetuning/blob/main/figures/FSGM_comparison.png)


After that, we finetuned the pretrained checkpoints on CIFAR10 and we investigated through a variety of plots whether the robustness of the initial checkpoints had an impact on the finetuning process, by measuring the generalization error as well as the distance travelled in weight space. We found that, consistently across runs, the checkpoint that was most robust had to travel the least in weight space to achieve good finetuning performance. As for the generalization error, we did not observe a significant correlation: observations are quite noisy and error bars overlap.

Distance travelled             |  Generalization performance
:-------------------------:|:-------------------------:
![](https://github.com/MattiaSC01/Robustness-of-Minima-and-Finetuning/blob/main/figures/distance_travelled.png)  |  ![](https://github.com/MattiaSC01/Robustness-of-Minima-and-Finetuning/blob/main/figures/finetuning_losses.png)
