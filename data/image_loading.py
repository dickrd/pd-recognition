from PIL import Image

def load_compcar_with_crop(image_path, resize=(512, 512)):
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


def load_image_data(image_path, resize=(512, 512)):
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


def load_car_json_data(car_json_path, resize=(512, 512),
                       crop_percentage=1.0, name_filter=None):
    """
    Load car json file.
    :param car_json_path: Path to a car json file.
    :param name_filter: A list of desired car models. If provided, all other models will be discarded.
    :param resize: Resize the car image to the size.
    :param crop_percentage: Crop operation will try to maintain this much percentage of the original image.
    :return: PIL.Image object of the car's image and the label(car brand).
    """
    import json
    import base64
    import io
    from PIL import Image

    with open(car_json_path, 'r') as json_data:
        j = json.load(json_data)

        # Label of this image.
        label = j["brand"].split("#")[1]

        if name_filter:
            import re

            model = j["brand"].lower()
            pattern = re.sub(r"\(.+\)", "", model).strip()

            found = False
            for item in name_filter:
                target = re.sub(r"\(.+\)", "", item.lower()).strip()
                if target in pattern:
                    found = True
                    print "Found: " + model.encode("utf-8")
                    break
            if not found:
                return None, None

        # Image bytes.
        image_bytes = base64.b64decode(j["image"])

        # Pre-process image.
        image = Image.open(io.BytesIO(image_bytes)).convert(mode="RGB")

        # Crop image.
        cropped = (1.0 - crop_percentage) / 2.0
        original_width = image.size[0]
        original_height = image.size[1]
        left = original_width * cropped
        upper = original_height * cropped
        image = image.crop((left, upper, original_width - left, original_height - upper))

        # Resize image.
        image = image.resize(resize, Image.LANCZOS)

    return image, label
