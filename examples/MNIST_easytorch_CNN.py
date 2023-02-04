import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from examples.models import MNISTNet

from easytorch import EasyTorch, ETRunner, ConfusionMatrix, ETMeter, AUCROCMetrics

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class MNISTTrainer(ETRunner):
    def _init_nn_model(self):
        self.nn['model'] = MNISTNet()

    def iteration(self, batch):
        inputs = batch[0].to(self.device['gpu']).float()
        labels = batch[1].to(self.device['gpu']).long()

        out = self.nn['model'](inputs)
        loss = F.nll_loss(out, labels)

        max_prob, pred = torch.max(out, 1)

        meter = self.new_meter()
        meter.averages.add(loss.item(), len(inputs))

        # Add extra loss if needed as:
        meter.averages.add(loss.item() * 0.3, len(inputs), index=1)

        meter.metrics['cmf'].add(pred, labels.float())
        meter.metrics['auc'].add(max_prob, labels.float())

        return {'loss': loss, 'meter': meter, 'predictions': pred}

    def init_cache(self):
        self.cache['log_header'] = 'Loss1,Loss2|Accuracy,F1,Precision,Recall'
        self.cache.update(monitor_metric='f1', metric_direction='maximize')

    def new_meter(self):
        return ETMeter(
            num_averages=2,  # Since we are tracing two losses
            cmf=ConfusionMatrix(num_classes=10),
            auc=AUCROCMetrics()
        )


train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)
val_dataset = datasets.MNIST('../data', train=False,
                             transform=transform)

dataloader_args = {'train': {'dataset': train_dataset},
                   'validation': {'dataset': val_dataset}}
runner = EasyTorch(phase='train', distributed_validation=True,
                   batch_size=512, epochs=21,
                   dataloader_args=dataloader_args,
                   image_size=(28, 28))

if __name__ == "__main__":
    runner.run(MNISTTrainer)
