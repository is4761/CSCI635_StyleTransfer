################################################################
# Style_Transfer.py
# 
# CSCI 635 group 6
# 
# This Python script enables neural style transfer using the VGG19 neural network architecture. 
# It takes a content image and a style image as inputs and generates a new image 
# that combines the content of the content image with the style of the style image.
################################################################
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import optim as optim
import requests
from torchvision import transforms, models

# Constants
CONTENT_IMG = 'images/octopus.jpg'
STYLE_IMG = 'images/hockney.jpg'
SHOW_EVERY = 5 # for displaying the target image, intermittently
STEPS = 5000
LEARNING_RATE = 0.003
CONTENT_WEIGHT = 1  # alpha
BETAS = [1e6]  # beta # switch with 1, 1e3, 1e6

# Load the model
vgg = models.vgg19(weights=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)


def load_image(img_path, max_size=400, shape=None):
    """Load in and preprocess the image, making sure the image is <= 400 pixels."""
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image

def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation output
                  '28': 'conv5_1'}

    features = {}
    x = image
    
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    # get the batch_size, depth, height, and width of the Tensor
    batch_size, d, h, w = tensor.size()

    # reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())  # multiply with transpose

    return gram


# load in content and style image
content = load_image(CONTENT_IMG).to(device)
style = load_image(STYLE_IMG, shape=content.shape[-2:]).to(device)

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

# get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in
               style_features}  # dic to store layer name : gram matrix

target = content.clone().requires_grad_(True).to(device)

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

# iteration hyperparameters
optimizer = optim.Adam([target], lr=LEARNING_RATE)
steps = 5000  # decide how many iterations to update your image (5000)

for beta in BETAS:
    total_loss = 1
    ii=1
    while total_loss > 0:
        # keep track of iterations
        print(ii)

        ii = ii + 1
        # get the features from your target image
        target_features = get_features(target, vgg)

        # Then calculate the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # initialize the style loss to 0
        style_loss = 0

        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]

            # Calculate the target gram matrix
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape

            # get the "style" style representation
            style_gram = style_grams[layer]

            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate the *total* loss
        total_loss = CONTENT_WEIGHT * content_loss + beta * style_loss
        if total_loss < 1:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.axis('off')
            plt.title('Final image Iteration '+str(ii)+' loss = '+str(total_loss))
            file_name = 'iteration_'+str(ii)
            plt.savefig(file_name+'.png')
        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # display intermediate images and print the loss
        if ii % SHOW_EVERY == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.axis('off')
            plt.title('Final image Iteration '+str(ii)+' loss = '+str(total_loss))
            file_name = 'iteration_'+str(ii)
            plt.savefig(file_name+'.png')
