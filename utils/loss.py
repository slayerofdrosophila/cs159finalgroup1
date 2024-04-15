from torch.nn import CrossEntropyLoss


class LossCalculator(object):
    def __init__(self):
        self.criterion = CrossEntropyLoss()
        self.loss_seq = []
    
    def calculate_loss(self, output, target):
        """Calculates the loss of the model based on the output and target labels, using
        the CrossEntropyLoss criterion."""

        loss = self.criterion(output, target)        
        self.loss_seq.append(loss.item())
        return loss

    def get_average_loss(self, past_records=100):
        """Returns the average loss of the past n records."""

        past_records = min(past_records, len(self.loss_seq))
        return sum(self.loss_seq[-past_records:]) / past_records


def compute_accuracy(output, target):
    """Computes the accuracy of the model based on the output and target labels."""

    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    correct = pred.eq(target.view(-1, 1)).sum().item()

    accuracy = correct * 100.0 / batch_size
    return accuracy
