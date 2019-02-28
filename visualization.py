import matplotlib.pyplot as plt
import numpy as np
from dataloaders import mean, std, load_cifar10
import torchvision
from torch import nn
from torchvision.utils import save_image

def save_original_image(normalized):
    orig =  normalized.numpy()
    orig = orig.T * std + mean
    orig = np.transpose(orig.T, (1, 2, 0))
    plt.imsave("./visualizations/original_image.png", orig)

model = torchvision.models.resnet18(pretrained=True)
dataloader, _, _ = load_cifar10(32)
image = next(iter(dataloader))[0][0]
save_original_image(image)
image = image.view(1, *image.shape)
image = nn.functional.interpolate(image, size=(256,256))

first_layer_out = model.conv1(image)
to_visualize = first_layer_out.view(first_layer_out.shape[1], 1, *first_layer_out.shape[2:])
save_image(to_visualize, "./visualizations/filters_first_layer.png")

last_conv_layer_out = image
for name, layer in model.named_children():
    if name == 'avgpool' or name == 'fc':
        break
    last_conv_layer_out = layer(last_conv_layer_out)

last_conv_img = last_conv_layer_out.view(last_conv_layer_out.shape[1], 1, *last_conv_layer_out.shape[2:])[:32]
save_image(last_conv_img, "./visualizations/filters_last_conv_layer.png")

save_image(model.conv1.weight.data, "./visualizations/weights.png")