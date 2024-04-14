import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils.data import get_dataloader
from utils.loss import LossCalculator, compute_accuracy
from utils.model import get_model

from utils import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_network(model=get_model(), write=True):
    writer = SummaryWriter() if write else None

    model = model.to(device)
    dataloader = get_dataloader()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=0.5, last_epoch=args.resume_epoch)
    loss_calc = LossCalculator()

    for epoch in range(args.resume_epoch + 1, args.epochs):
        time_start = time.time()
        accuracy, loss = train_step(model, dataloader, loss_calc, optimizer, scheduler)
        time_end = time.time()
        print(f"Epoch {epoch + 1} ({time_end - time_start}), \t Accuracy: {accuracy}, \t Loss: {loss}")

        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'loss_sequence': loss_calc.loss_seq},
                   args.save_path)
        if writer is not None:
            writer.add_scalar("training/accuracy", accuracy, epoch)
            writer.add_scalar("training/loss", loss_calc.get_average_loss(), epoch)

    if writer is not None:
        writer.close()


def train_step(model, dataloader, loss_calc, optimizer, scheduler):
    model.train()
    torch.backends.cudnn.benchmark = True  # set benchmark flag to faster runtime

    total_accuracy = 0

    for iteration, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = loss_calc.calculate_loss(outputs, targets)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        total_accuracy += compute_accuracy(outputs.data, targets)

    return total_accuracy / len(dataloader), loss_calc.get_average_loss()
