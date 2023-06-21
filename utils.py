import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, efficientnet_v2_s

import numpy as np
import matplotlib.pyplot as plt

import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataloader, model, loss_fn, optimizer, log=100, std=None):
    size = len(dataloader.dataset)  # the .dataset attribute gives us the original data, so this is 60000
    # the dataloader is an iterable that produces batchs, so its length is the num. of batches
    num_batches = len(dataloader)
    model.train()  # set the model in training mode (important for the batchnorm and dropout layers)
    mean_loss = 0.0
    correct = 0

    loss_no_noise = 0
    acc_no_noise = 0
    for batch, (X, y) in enumerate(dataloader):

        # send the data to the device. This is crucial if we're using the GPU
        X, y = X.to(device), y.to(device)

        if std is not None:
            with torch.no_grad():
                pred = model(X)
                loss_no_noise += loss_fn(pred, y)
                acc_no_noise += (pred.argmax(1) == y).type(torch.float).sum().item()

        # add noise to pictures - improve robustness
        if std is not None:
            X = X + std * torch.randn_like(X)
            X = torch.clamp(X, 0, 1)

        ## Forward pass.
        ## This computes the loss but it also builds the computational graph,
        ##   storing it in a distributed fashion in the various tensors involved.
        ## The loss is a tensor with a single entry, not a float. This is crucial for backpropagation.
        pred = model(X)  # pass thorugh the model; `pred` has the logits
        loss = loss_fn(pred, y)  # compute the loss; note that `y` will be translated into a 1-hot-encoded vactor

        ## Backward pass.
        optimizer.zero_grad()  # reset the gradients inside all tensors that were given as parameters to the optimizer
        loss.backward()  # trigger the backward pass; here all the gradients are computed; each tensor stores its own
        optimizer.step()  # apply one step of optimization to the parameters, using the gradient information

        ## Here we just keep track of the loss and error to monitor the training.
        ## Note that the `item()` method gives you the value of a single-valued tensor, such as the loss.
        mean_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        ## Some crude visual feedback to display our progress
        if batch % log == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"  [{current:>5d}/{size:>5d}]")

    ## Compute and print some data to monitor the training
    mean_loss /= num_batches
    correct /= size

    if std is not None:
        loss_no_noise /= num_batches
        acc_no_noise /= size

    # print(f"TRAINING - Accuracy: {(100 * correct):>5.1f}%, Avg loss: {mean_loss:>7f}")
    if std is None:
        return mean_loss, correct
    else:
        return mean_loss, correct, loss_no_noise, acc_no_noise


def test(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    mean_loss = 0.0
    correct = 0

    with torch.no_grad():
        for (img, label) in dataloader:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            mean_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        mean_loss /= num_batches
        correct /= size

    return mean_loss, correct


def load_cifar(num_classes, size=(112, 112)):
    # data preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
    ])

    if num_classes == 100:
        # Load training dataset
        train_set = torchvision.datasets.CIFAR100(
            root='./CIFAR10/train_set',
            train=True,
            download=True,
            transform=transform
        )

        # Load test set
        test_set = torchvision.datasets.CIFAR100(
            root="./CIFAR10/test_set",
            train=False,
            download=True,
            transform=transform,
        )

    elif num_classes == 10:
        # Load training dataset
        train_set = torchvision.datasets.CIFAR10(
            root='./CIFAR10/train_set',
            train=True,
            download=True,
            transform=transform
        )

        # Load test set
        test_set = torchvision.datasets.CIFAR10(
            root="./CIFAR10/test_set",
            train=False,
            download=True,
            transform=transform,
        )

    else:
        raise Exception(f'CIFAR{num_classes} not found!')

    return train_set, test_set


# TODO: compute also accuracy
def denoising(model, dataloader, loss_fn, noise, iters, max_b=None, ):
    model.eval()  # eval mode

    if max_b is None:
        max_b = len(dataloader)

    if max_b > len(dataloader):
        raise Exception('max_b parameter must be less than or equal to the total number of batches!')

    losses = {}
    accs = {}
    with torch.no_grad():
        
        for stddev in noise:
            losses[stddev] = 0
            accs[stddev] = 0
        # repeatedly denoise noisy inputs
        size = 0
        for idx, (img, label) in enumerate(dataloader):
            size += len(img)
            for stddev in noise:
            
            
                for i in range(iters):
                # compute loss using noisy inputs
                    img, label = img.to(device), label.to(device)


                    noisy = torch.clamp(img + torch.randn_like(img) * stddev, 0, 1)

                    pred = model(noisy)

                    losses[stddev] += loss_fn(pred, label)
                    accs[stddev] += (pred.argmax(1) == label).type(torch.float).sum().item()

                    # do exactly max_b batches
                    if idx + 1 >= max_b:
                        break
        
        for stddev in noise:
            losses[stddev] /= (max_b*iters)
            losses[stddev] = losses[stddev].cpu()
            accs[stddev] /= (size*iters)
                
#                 losses[stddev].append(loss.cpu() / max_b)
#                 accs[stddev].append(acc / size)

    return losses, accs

def test_fgsm(test_loader, model, epsilon, max_b, loss_fn):
    # Accuracy counter
    correct = 0
    losses = 0
    size = 0
    if max_b is None:
        max_b = len(test_loader)
    # Loop over all examples in test set
    for idx, (data, label) in enumerate(test_loader):
#         print(idx, len(test_loader))
        
        size += len(data)
        # Send the data and label to the device
        data, label = data.to(device), label.to(device)
        
        model.zero_grad()
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        # Calculate the loss
        loss = loss_fn(output, label)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()


        # Call FGSM Attack
#       data = fgsm_attack(data, epsilon, data.grad.data)
        
        perturbed_data = data + epsilon * data.grad.data_fgsm.sign()
        # Adding clipping to maintain [0,1] range
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        with torch.no_grad():
            model.eval()
            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            losses += loss_fn(output, label)
            correct += (output.argmax(1) == label).type(torch.float).sum().item()

        if idx + 1 > max_b:
            break

    # Calculate final accuracy for this epsilon
    final_acc = correct / size
    final_loss = losses.cpu() / max_b
    # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {size} = {final_acc}\tAverage loss: {final_loss}")

    # Return the accuracy and an adversarial example
    return final_acc, final_loss


# TODO: check efficiency (switch order of loops)
def weight_flatness(model, dataloader, loss_fn, noise, iters, max_b=None, chkpt='checkpoint.pt'):
    # save model checkpoint here (path with .pt extension)

    model.eval()  # eval mode

    if max_b is None:
        max_b = len(dataloader)

    if max_b > len(dataloader):
        raise Exception('max_b parameter must be less than or equal to the total number of batches!')

    # save results
    losses = {}
    accs = {}

    # save current weights
    torch.save(model.state_dict(), chkpt)

    with torch.no_grad():

        # repeatedly perturb model weights starting from initial configuration
        for stddev in noise:
            print(f"Computing flatness with {stddev=}")
            losses[stddev] = []
            accs[stddev] = []

            for i in range(iters):

                # get model weights as an Ordered Dictionary
                d = model.state_dict()

                # add noise to each Tensor (magnitude-aware)
                for key in d:
                    if not ('conv' in key or 'fc' in key):
                        continue
                    noise = torch.randn_like(d[key]) * stddev
                    d[key] *= (1 + noise)

                # load updated weights into the model
                model.load_state_dict(d)

                # compute loss with perturbed weights
                loss = 0
                size = 0
                acc = 0

                for idx, (img, label) in enumerate(dataloader):
                    img, label = img.to(device), label.to(device)
                    size += len(img)

                    pred = model(img)
                    loss += loss_fn(pred, label)
                    acc += (pred.argmax(1) == label).type(torch.float).sum().item()

                    if idx + 1 >= max_b:
                        break

                losses[stddev].append(loss.cpu() / max_b)
                accs[stddev].append(acc / size)

                # reset weights to their initial values
                model.load_state_dict(torch.load(chkpt))

    return losses, accs


# assume batches are (B, C, W, H)
def mask(dataloader, test: callable, dx=10, dy=10, black=None):
    if black is None:
        black = 0

    res = []

    for (img, label) in dataloader:
        B, C, W, H = img.shape
        x, y = np.randint(W - dx), np.randint(H - dy)

        img[:, :, x: x + dx, y: y + dy] = black

        res.append(test(img, label))

    return res


def test_builder():
    def test(img, label):
        pass

    return test


# distance: what to return? normalize by number of parameters? return a separate distance for convolutions and dense only?
def traverse(dataloader, model1, model2, loss_fn, nsteps, chkpt='chkpt.pt'):
    model1.eval()
    model2.eval()

    d1, d2 = model1.state_dict(), model2.state_dict()

    # save weights of model1 to restore them later
    torch.save(d1, chkpt)

    with torch.no_grad():

        # compute weight difference
        diff = {key: d2[key] - d1[key] for key in d1}

        l2_distance, l2_convs, l2_bn, count, convs_count, bn_count = 0, 0, 0, 0, 0, 0

        for key in diff:

            # compute distance and number of params
            d = (diff[key] ** 2).sum()
            n = torch.numel(diff[key])

            # update counts
            l2_distance += d
            count += n

            if 'conv' in key or 'fc' in key:
                l2_convs += d
                convs_count += n
            else:
                l2_bn += d
                bn_count += n

        # take sqrt and normalize by number of params
        l2_distance, l2_convs, l2_bn = torch.sqrt(l2_distance), torch.sqrt(l2_convs), torch.sqrt(l2_bn)
        l2_distance /= count
        l2_convs /= convs_count
        l2_bn /= bn_count

        # store losses
        losses = []
        loss = test(dataloader, model1, loss_fn)
        losses.append(loss)

        for i in range(nsteps):

            # move model1 in weight space towards model2
            for key in d1:
                if 'num_batches_tracked' in key:
                    d1[key] += diff[key] // nsteps
                else:
                    d1[key] += diff[key] / nsteps

            # load new weights
            model1.load_state_dict(d1, strict=True)

            # compute loss
            loss = test(dataloader, model1, loss_fn)
            losses.append(loss)

    # restore initial weights of model1
    model1.load_state_dict(torch.load(chkpt))

    return losses, l2_distance, l2_convs, l2_bn
