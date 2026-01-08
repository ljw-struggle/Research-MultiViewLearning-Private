import os, copy, time, argparse
import numpy as np
import scipy.ndimage as nd
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

exp_evidence = lambda y: torch.exp(torch.clamp(y, -10, 10))
relu_evidence = lambda y: F.relu(y)
softplus_evidence = lambda y: F.softplus(y)

class LeNet(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) if self.use_dropout else x
        x = self.fc2(x)
        return x
    
############################################################################################################################################################################

def kl_divergence(alpha, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True)))
    second_term = ((alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True))
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device); alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32)) # scaler tensor does not have .to(device)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step): 
    def loglikelihood_loss(y, alpha):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = y.to(device); alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device); alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha)
    annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div

# Loss functions for uncertainty
def edl_log_loss(output, target, epoch_num, num_classes, annealing_step): # Corresponding to the equation 3 in the paper
    evidence = relu_evidence(output); alpha = evidence + 1
    return torch.mean(edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step))

def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step): # Corresponding to the equation 4 in the paper
    evidence = relu_evidence(output); alpha = evidence + 1
    return torch.mean(edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step))

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step): # Corresponding to the equation 5 in the paper
    evidence = relu_evidence(output); alpha = evidence + 1
    return torch.mean(mse_loss(target, alpha, epoch_num, num_classes, annealing_step))

############################################################################################################################################################################

def train_model(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs=25, device=None, uncertainty=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            print("Training...") if phase == "train" else print("Validating...")
            running_loss = 0.0; running_corrects = 0.0; correct = 0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device); labels = labels.to(device) # inputs: [batch_size, 1, 28, 28], labels: [batch_size]
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"): # track history if only in train
                    if uncertainty:
                        one_hot_embedding = lambda labels, num_classes: torch.eye(num_classes).to(device)[labels]
                        y = one_hot_embedding(labels, num_classes).to(device) # shape: [batch_size, num_classes]
                        outputs = model(inputs) # shape: [batch_size, num_classes]
                        _, preds = torch.max(outputs, 1) # shape: [batch_size]
                        loss = criterion(outputs, y.float(), epoch, num_classes, 10)
                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1)) # shape: [batch_size, 1]
                        acc = torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward(); optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if scheduler is not None:
                scheduler.step() if phase == "train" else None
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            losses["loss"].append(epoch_loss); losses["phase"].append(phase); losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item()); accuracy["epoch"].append(epoch); accuracy["phase"].append(phase)
            print("{} loss: {:.4f} acc: {:.4f}".format(phase.capitalize(), epoch_loss, epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # deep copy the model, otherwise the model will be overwritten during training
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    model.load_state_dict(best_model_wts) # load best model weights
    metrics = (losses, accuracy)
    return model, metrics

def test_single_image(model, img_path, uncertainty=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path).convert("L") # Convert to grayscale image
    num_classes = 10
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)
    img_variable = img_tensor.clone().detach().to(device)
    if uncertainty:
        output = model(img_variable)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)
        print("Uncertainty:", uncertainty)
    else:
        output = model(img_variable)
        _, preds = torch.max(output, 1)
        prob = F.softmax(output, dim=1)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)
    labels = np.arange(10)
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})
    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))
    axs[0].set_title("One")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")
    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")
    fig.tight_layout()
    plt.savefig("./results/{}".format(os.path.basename(img_path)))

def rotating_image_classification(model, img, filename, uncertainty=False, threshold=0.5, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        rotate_img = lambda x, deg: nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        rimgs[:, i * 28 : (i + 1) * 28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = img_tensor.clone().detach().to(device)
        if uncertainty:
            output = model(img_variable)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            _, preds = torch.max(output, 1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())
            lu.append(uncertainty.mean().cpu().detach().numpy())
        else:
            output = model(img_variable)
            _, preds = torch.max(output, 1)
            prob = F.softmax(output, dim=1)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())
        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.tolist())
    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"] * 2
    labels = labels.tolist()
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})
    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])
    if uncertainty:
        labels += ["uncertainty"]
        axs[2].plot(ldeg, lu, marker="<", c="red")
    print(classifications)
    axs[0].set_title('Rotated "1" Digit Classifications')
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")
    plt.pause(0.001)
    empty_lst = [classifications]
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")
    axs[2].legend(labels)
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")
    plt.savefig(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int, help="Desired number of epochs.")
    parser.add_argument("--dropout", action="store_true", help="Whether to use dropout or not.")
    parser.add_argument("--uncertainty", action="store_true", help="Use uncertainty or not.")
    mode_group = parser.add_mutually_exclusive_group(required=True) # Running mode
    mode_group.add_argument("--train", action="store_true", help="To train the network.")
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument("--examples", action="store_true", help="To example MNIST data.")
    untype_group = parser.add_mutually_exclusive_group() # Uncertainty type
    untype_group.add_argument("--digamma", action="store_true", help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.")
    untype_group.add_argument("--log",action="store_true", help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.")
    untype_group.add_argument("--mse", action="store_true", help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.")
    args = parser.parse_args()
    parser.error("Must specify one runnning mode.") if sum([args.train, args.test, args.examples]) != 1 else None
    parser.error("Must specify one uncertainty type when using uncertainty.") if args.uncertainty and sum([args.digamma, args.log, args.mse]) != 1 else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_train = MNIST("./data/mnist", download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    data_val = MNIST("./data/mnist", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    dataloader_train = DataLoader(data_train, batch_size=1000, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)
    dataloaders = {"train": dataloader_train, "val": dataloader_val}
    
    if args.train: # Train the model
        model = LeNet(dropout=args.dropout).to(device)
        if args.uncertainty and args.digamma:
            criterion = edl_digamma_loss
        if args.uncertainty and args.log:
            criterion = edl_log_loss
        if args.uncertainty and args.mse:
            criterion = edl_mse_loss
        if not args.uncertainty:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        model, metrics = train_model(model, dataloaders, 10, criterion, optimizer, scheduler=exp_lr_scheduler, num_epochs=args.epochs, device=device, uncertainty=args.uncertainty)
        state = {"epoch": args.epochs, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
        if args.uncertainty and args.digamma:
            torch.save(state, "./results/model_uncertainty_digamma.pt"); print("Saved: ./results/model_uncertainty_digamma.pt")
        if args.uncertainty and args.log:
            torch.save(state, "./results/model_uncertainty_log.pt"); print("Saved: ./results/model_uncertainty_log.pt")
        if args.uncertainty and args.mse:
            torch.save(state, "./results/model_uncertainty_mse.pt"); print("Saved: ./results/model_uncertainty_mse.pt")
        if not args.uncertainty:
            torch.save(state, "./results/model.pt"); print("Saved: ./results/model.pt")
    
    if args.test: # Test the model
        digit_one, _ = data_val[5] # 5 is the index of the digit "1" in the MNIST dataset, _ is the target
        model = LeNet().to(device)
        optimizer = optim.Adam(model.parameters())
        if args.uncertainty and args.digamma:
            checkpoint = torch.load("./results/model_uncertainty_digamma.pt"); filename = "./results/rotate_uncertainty_digamma.jpg"
        if args.uncertainty and args.log:
            checkpoint = torch.load("./results/model_uncertainty_log.pt"); filename = "./results/rotate_uncertainty_log.jpg"
        if args.uncertainty and args.mse:
            checkpoint = torch.load("./results/model_uncertainty_mse.pt"); filename = "./results/rotate_uncertainty_mse.jpg"
        if not args.uncertainty:
            checkpoint = torch.load("./results/model.pt"); filename = "./results/rotate.jpg"
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.eval()
        rotating_image_classification(model, digit_one, filename, uncertainty=args.uncertainty)
        test_single_image(model, "./data/one.jpg", uncertainty=args.uncertainty)
        test_single_image(model, "./data/yoda.jpg", uncertainty=args.uncertainty)
    
    if args.examples: # Display examples
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")
        