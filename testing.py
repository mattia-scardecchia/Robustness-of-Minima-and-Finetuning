import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# compute the loss and accuracy on a test set

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


# evaluate a callable test on randomly masked inputs

def test_with_random_masking(dataloader, test: callable, dx=10, dy=10, black=None):
    if black is None:
        black = 0

    res = []

    for (img, label) in dataloader:
        B, C, W, H = img.shape
        x, y = np.randint(W - dx), np.randint(H - dy)

        img[:, :, x: x + dx, y: y + dy] = black

        res.append(test(img, label))

    return res


# Traverse the weight space along the line connecting two models in small discrete steps, 
# computing the loss and accuracy on a test set at each step.

def traverse(dataloader, model1, model2, loss_fn, nsteps, save_path='chkpt.pt'):
    model1.eval()
    model2.eval()

    d1, d2 = model1.state_dict(), model2.state_dict()

    # save weights of model1 to restore them later
    torch.save(d1, save_path)

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
    model1.load_state_dict(torch.load(save_path))

    return losses, l2_distance, l2_convs, l2_bn


# compute the local energy of a model on a test set

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


# Compute the loss and accuracy on a test set, using adversarial examples generated with FGSM.

def test_fgsm(test_loader, model, epsilon, max_b, loss_fn):
    
    # Accuracy counter
    correct = 0
    losses = 0
    size = 0
    if max_b is None:
        max_b = len(test_loader)

    # Loop over all examples in test set
    for idx, (data, label) in enumerate(test_loader):
        
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
        
        # Collect datagrad
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

    return final_acc, final_loss


# compute the loss and accuracy on a test set, adding gaussian noise to the inputs

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

    return losses, accs
