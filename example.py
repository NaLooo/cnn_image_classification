from models import CNN
from trainer import Trainer

model = CNN()
trainer = Trainer(model, 'fashion_mnist')

trainer.train(model, 'fashion_mnist', epochs=5)
trainer.evaluate()