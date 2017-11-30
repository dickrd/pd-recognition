#coding=utf-8
from PIL import Image


def get_label_path_of_compcar(image_path):
    import re
    return re.sub(r"(.*)/image(/|$)", r"\1/label\2", image_path)


def load_compcar_with_crop(image_path, resize=(512, 512)):
    # Load image data.
    image = Image.open(image_path).convert(mode="RGB")

    # Read label.
    label_path = get_label_path_of_compcar(image_path)
    with open(label_path[:-3] + "txt", 'r') as label_file:
        label_file.readline()
        label_file.readline()
        line = label_file.readline()

        x1, y1, x2, y2 = map(int, line.strip().split(" "))
    # Crop image.
    image = image.crop((x1, y1, x2, y2))

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

        # Name of this image.
        name = j["brand"].split("#")[1]

        if name_filter:
            import re

            # Full qualified name of this image.
            model = j["brand"].lower()
            pattern = re.sub(r"\(.+\)", "", model).strip()

            # Search in name filter.
            found = False
            for name in name_filter:
                approved_name = re.sub(r"\(.+\)", "", name.lower()).strip()
                # If the name of this image is approved.
                if approved_name in pattern:
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

    return image, name

def load_car_color_json_data(car_json_path, resize=(512,512),
                             crop_percentage=1.0, name_filter=None):
    """
    load car json file.
    ：param car_json_path: Path to a car json file.
    : param resize: Resize image to this size.
    : return: PIL.Image object of the car's image and the label(car color).
    """

    if name_filter is not None:
        raise NotImplementedError("Does not support filter")
    if crop_percentage != 1.0:
        raise NotImplementedError("Does not support crop")

    import json
    import base64
    import io
    from PIL import Image
    color_list = [u"白", u"灰", u"黄", u"粉", u"红", u"紫", u"绿", u"蓝", u"棕", u"黑"]
    with open(car_json_path,'r') as json_data:
        j = json.load(json_data)
        #Color of this image
        color = j["color"]
        if color:
            color = color.split("#")[0]
            #color = color.decode('utf-8')
            if color[-1] in color_list:
                color = color[-1]
            elif color[-1] == u"银":
                color = u"灰"
            else:
                flag = True
                for v in color:
                    if v in color_list:
                        color = v
                        flag = False
                if flag:
                    color =u"其他"

            image_bytes = base64.b64decode(j["image"])
            image = Image.open(io.BytesIO(image_bytes)).convert(mode="RGB")
            image = image.resize(resize,Image.LANCZOS)
        else:
            return None,None

    return image, color
