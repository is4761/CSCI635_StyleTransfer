# Neural Style Transfer with VGG19

This Python script enables neural style transfer using the VGG19 neural network architecture. It takes a content image and a style image as inputs and generates a new image that combines the content of the content image with the style of the style image.

## Prerequisites
- Python 3.x
- PyTorch 2.2.2
- torchvision 0.17.2
- matplotlib 3.6.2
- PIL 10.3.0
- numpy 1.26.4

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/is4761/CSCI635_StyleTransfer.git
   ```

2. Install the required dependencies:
   ```bash
   conda config --add channels conda-forge
   conda config --add channels pytorch
   conda install matplotlib==3.6.2
   conda install numpy==1.26.4
   conda install Pillow==10.3.0
   conda install Requests==2.31.0
   conda install pytorch==2.2.2
   conda install torchvision==0.17.2
   ```

## Usage
1. Prepare your content and style images. You can use images from the images folder provided or your local machine.
2. Update the `CONTENT_IMG` and `STYLE_IMG` variables in the script with the paths of your content and style images.
3. Adjust any other hyperparameters or settings in the script according to your preferences.
4. Run the script:
   ```bash
   python Style_Transfer.py
   ```
   The script will generate intermediate images during the optimization process and save them as `iteration_X.png`, where X is the iteration number.
5. Once you feel that the total loss is converging, generate a keyboard exception to stop the program.

## Parameters
- `CONTENT_IMG`: Path or URL of the content image.
- `STYLE_IMG`: Path or URL of the style image.
- `max_size`: Maximum size of the input images (default: 400).
- `style_weights`: Weights for style loss at different layers of the VGG19 network.
- `content_weight`: Weight for content loss.
- `betas`: List of weights for style loss (beta values).
- `show_every`: Frequency of displaying intermediate images during optimization.

## References
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
