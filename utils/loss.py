from torch.nn import CrossEntropyLoss


class LossCalculator(object):
    def __init__(self):
        self.criterion = CrossEntropyLoss()
        self.loss_seq = []
    
    def calculate_loss(self, output, target):
        loss = self.criterion(output, target)        
        self.loss_seq.append(loss.item())
        return loss

    def get_average_loss(self, past_records=100):
        past_records = min(past_records, len(self.loss_seq))
        return sum(self.loss_seq[-past_records:]) / past_records


# def compute_accuracy(output, target, topk=(1,)):
#     """
#         Computes the precision@k for the specified values of k
#         ref: https://github.com/chengyangfu/pytorch-vgg-cifar10
#     """
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)  # used to be .view
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def compute_accuracy(output, target):
    """
    Computes the accuracy for k=1
    """
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    correct = pred.eq(target.view(-1, 1)).sum().item()

    accuracy = correct * 100.0 / batch_size
    return accuracy
