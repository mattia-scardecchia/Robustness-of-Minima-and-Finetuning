import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# train a model for one epoch. Supports training with noisy data.

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


# compute the normalized euclidean distance between two weight configurations

def compute_euclidean_distance(current_parameters, initial_parameters):
    par_0 = initial_parameters
    par_t = current_parameters
    M = sum([torch.numel(par) for par in par_0])

    with torch.no_grad():
        corr = 0
        for i in range(len(par_0)):
            w_0 = torch.Tensor(par_0[i])
            w_t = torch.Tensor(par_t[i])

            square = (w_0 - w_t) ** 2
            corr += torch.sum(square)

        result = corr / M
        return result
    

# call compute_euclidean_distance and log to wandb

def handle_distance_computation(current_parameters, initial_parameters, t, distances):
    with torch.no_grad():
        if t >= 0:

            distance = compute_euclidean_distance(current_parameters, initial_parameters)
            distances.append(float(distance))
            wandb.log({'Correlation Function wrt initial parameters': distance}, step=t)

        else:
            distances.append('\\')


# Used for finetuning. Train the model for one epoch, and compute the euclidean distance between the current weights and the initial weights.

def train_with_correlation_computation(dataloader, model, loss_fn, optimizer, initial_parameters, distances, epoch, ALL_par):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    mean_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Step [{batch + 1}/{num_batches}], Loss {loss:.4f}")

        if (batch + 1) % 100 == 0:
            step = (num_batches * epoch) + (batch + 1)
            current_parameters = []

            with torch.no_grad():

                for name, param in model.named_parameters():
                    if ALL_par:
                        if ('conv' in name) or ('fc' in name):
                            current_parameters.append(param.clone())
                    else:
                        current_parameters.append(param.clone())

                handle_distance_computation(current_parameters, initial_parameters, step, distances)
                print(f"Decorrelation from initial weights : {distances[-1]}")

    mean_loss /= num_batches
    correct /= size
    print(f" \nTRAINING - Accuracy: {(100 * correct):>5.1f}%, Avg loss: {mean_loss:>7f}")
    return mean_loss, correct
