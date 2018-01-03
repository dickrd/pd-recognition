#coding=utf-8
import os

from PIL import Image


class AdienceUtil(object):

    def __init__(self):
        self.parent = None
        self.image_path = None
        self.label = None

    def walk(self, input_path):
        fold_pattern = "fold_{0}_data.txt"
        image_pattern = "aligned/{0}/landmark_aligned_face.{1}.{2}"
        fold_count = 5
        fold_path = []

        if os.path.isdir(input_path):
            self.parent = input_path
            for index in range(fold_count):
                fold_path.append(os.path.join(input_path, fold_pattern.format(index)))
        else:
            self.parent = os.path.dirname(input_path)
            fold_path.append(input_path)

        for item in fold_path:
            try:
                with open(item, 'r') as fold_file:
                    for line in fold_file:
                        parts = line.split("\t")

                        # Skip header.
                        if parts[0] == "user_id":
                            continue

                        self.image_path = image_pattern.format(parts[0], parts[2], parts[1])
                        self.label = parts[3]

                        try:
                            age = int(self.label)
                            if age in range(0, 3):
                                self.label = "(0, 2)"
                            elif age in range(4, 7):
                                self.label = "(4, 6)"
                            elif age in range(8, 13):
                                self.label = "(8, 12)"
                            elif age in range(15, 21):
                                self.label = "(15, 20)"
                            elif age in range(25, 33):
                                self.label = "(25, 32)"
                            elif age in range(38, 44):
                                self.label = "(38, 43)"
                            elif age in range(48, 54):
                                self.label = "(48, 53)"
                            elif age >= 60:
                                self.label = "(60, 100)"
                        except ValueError:
                            if self.label == "(27, 32)":
                                self.label = "(25, 32)"
                            elif self.label == "(38, 42)":
                                self.label = "(38, 43)"
                            elif self.label == "(38, 48)":
                                self.label = "(38, 43)"

                        yield self.parent, None, [self.image_path]
            except IOError as e:
                print "Reading {0} failed: {1}".format(item, repr(e))

    # noinspection PyUnusedLocal
    def name(self, file_path, index=0):
        assert file_path == os.path.join(self.parent, self.image_path)
        return self.label


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
