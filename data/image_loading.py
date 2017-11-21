from PIL import Image

def load_compcar_with_crop(image_path,
                           resize=(224, 224)):
    import re

    # Load image data.
    image = Image.open(image_path).convert(mode="RGB")

    # Read label.
    label_path = re.sub(r"/image/?", r"/label/", image_path)
    with open(label_path, 'r') as label_file:
        label_file.readline()
        label_file.readline()
        line = label_file.readline()

        x1, y1, x2, y2 = line.strip().split(" ")
    # Crop image.
    image = image.crop((x1, y2, x2, y1))

    # Resize and return.
    image = image.resize(resize, Image.LANCZOS)
    return image


def load_image_data(image_path,
                    resize=(512, 512)):
    """
    Load image data in RGB format.
    :param image_path: Path to a image file.
    :param resize: Resize the car image to the size.
    :return: PIL.Image object of the car's image and the label(car brand).
    """

    # Load image data.
    image = Image.open(image_path).convert(mode="RGB")

    # Resize and return.
    image = image.resize(resize, Image.LANCZOS)
    return image
