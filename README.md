# CNN training for image classification
### Compatible with tensorflow and pytorch.
### Integrated dataset: mnist, fashion_mnist, cifar-10, cifar-100

## Quick Start
``` python
from models import CNN
from trainer import Trainer

model = CNN()
trainer = Trainer(model, 'fashion_mnist')

trainer.train(model, 'fashion_mnist', epochs=5)
trainer.evaluate()
```