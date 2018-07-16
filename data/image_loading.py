#coding=utf-8
import os

from PIL import Image


class AdienceUtil(object):

    def __init__(self, sex=False, regression=False):
        self.parent = None
        self.image_path = None
        self.label = None

        self.sex= sex
        self.regression = regression

    def _label_to_age_string(self):
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

    def _label_to_age_int(self):
        try:
            age = int(self.label)
            self.label = age
        except ValueError:
            if self.label == "(0, 2)":
                self.label = 1
            elif self.label == "(4, 6)":
                self.label = 5
            elif self.label == "(8, 12)":
                self.label = 10
            elif self.label == "(15, 20)":
                self.label = 18
            elif self.label == "(25, 32)":
                self.label = 28
            elif self.label == "(38, 43)":
                self.label = 40
            elif self.label == "(48, 53)":
                self.label = 50
            elif self.label == "(60, 100)":
                self.label = 80
            elif self.label == "(27, 32)":
                self.label = 30
            elif self.label == "(38, 42)":
                self.label = 39
            elif self.label == "(38, 48)":
                self.label = 45

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
                        if self.sex:
                            self.label = parts[4]
                        else:
                            self.label = parts[3]
                            if self.regression:
                                self._label_to_age_int()
                            else:
                                self._label_to_age_string()

                        yield self.parent, None, [self.image_path]
            except IOError as e:
                print "Reading {0} failed: {1}".format(item, repr(e))

    # noinspection PyUnusedLocal
    def name(self, file_path, index=0):
        assert file_path == os.path.join(self.parent, self.image_path)
        return self.label


class ImdbWikiUtil(object):

    def __init__(self, sex=False, regression=False, source_type="imdb"):
        self.parent = None
        self.image_path = None
        self.label = None

        self.sex= sex
        self.regression = regression
        self.source_type = source_type


    def _label_to_age_string(self):
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
        else:
            self.label = None


    def walk(self, input_path):
        from scipy.io import loadmat
        import re
        mat_file = loadmat(os.path.join(input_path, self.source_type + ".mat"))
        full_path = mat_file[self.source_type]['full_path'][0][0][0]
        gender = mat_file[self.source_type]['gender'][0][0][0]

        self.parent = input_path
        for img_path, gender_value in zip(full_path, gender):
            try:
                self.image_path = img_path[0]
                if self.sex:
                    if gender_value == 1:
                        self.label = 'male'
                    elif gender_value == 0:
                        self.label = 'female'
                    else:
                        self.label = None
                else:
                    self.label = int(self.image_path[-8:-4]) - int(re.search(r'(\d\d\d\d)-\d\d?-\d\d?', self.image_path).group(1))
                    if not self.regression:
                        self._label_to_age_string()

                yield self.parent, None, [self.image_path]
            except IOError as e:
                print "Processing {0} failed: {1}".format(img_path, repr(e))

    # noinspection PyUnusedLocal
    def name(self, file_path, index=0):
        assert file_path == os.path.join(self.parent, self.image_path)
        return self.label


class IogUtil(object):

    def __init__(self, sex=False, regression=False):
        self.parent = None
        self.image_path = None
        self.label = None
        self.position = None

        self.sex= sex
        self.regression = regression


    def walk(self, input_path):
        from scipy.io import loadmat
        import ntpath
        mat_file = loadmat(os.path.join(input_path, "eventrain.mat"))
        full_path = mat_file['trcoll']['name'][0][0][0]
        gender_col = mat_file['trcoll']['genClass'][0][0]
        age_col = mat_file['trcoll']['ageClass'][0][0]
        position_col = mat_file['trcoll']['facePosSize'][0][0]

        self.parent = input_path
        for img_path, gender, age, position in zip(full_path, gender_col, age_col, position_col):
            try:
                self.image_path = ntpath.basename(img_path[0])
                self.position = position
                if self.sex:
                    if gender[0] == 2:
                        self.label = 'male'
                    elif gender[0] == 1:
                        self.label = 'female'
                    else:
                        self.label = None
                else:
                    if self.regression:
                        self.label = int(age[0])
                    else:
                        self.label = str(age[0])

                yield self.parent, None, [self.image_path]
            except IOError as e:
                print "Processing {0} failed: {1}".format(img_path, repr(e))

    # noinspection PyUnusedLocal
    def name(self, file_path, index=0):
        assert file_path == os.path.join(self.parent, self.image_path)
        return self.label

    def load_img(self, file_path):
        assert file_path == os.path.join(self.parent, self.image_path)
        image = Image.open(file_path).convert(mode="RGB")

        base_measure = self.position[2] - self.position[0]
        left = self.position[4] - base_measure * 1.8
        top = self.position[5] - base_measure * 1.8
        return image.crop((left, top, left + base_measure * 3.6, top + base_measure * 3.6))


def normalize_image(image, resize=(512, 512), crop_percentage=1.0):
    """
    Resize and crop image object.
    :param image: PIL image object to process.
    :param resize: Resize the car image to the size.
    :param crop_percentage: Crop operation will try to maintain this much percentage of the original image.
    :return:
    """
    if crop_percentage < 1.0:
        # Crop image.
        cropped = (1.0 - crop_percentage) / 2.0
        original_width = image.size[0]
        original_height = image.size[1]
        left = original_width * cropped
        upper = original_height * cropped
        image = image.crop((left, upper, original_width - left, original_height - upper))

    # Resize image.
    image = image.resize(resize, Image.LANCZOS)

    return image


def get_label_path_of_compcar(image_path):
    import re
    return re.sub(r"(.*)/image(/|$)", r"\1/label\2", image_path)


def load_compcar_with_crop(image_path):
    # Load image data.
    image = Image.open(image_path).convert(mode="RGB")

    # Read label.
    label_path = get_label_path_of_compcar(image_path)
    with open(label_path[:-3] + "txt", 'r') as file_label:
        file_label.readline()
        file_label.readline()
        line = file_label.readline()

        x1, y1, x2, y2 = map(int, line.strip().split(" "))
    # Crop image.
    image = image.crop((x1, y1, x2, y2))

    return image


def load_image_data(image_path):
    """
    Load image data in RGB format.
    :param image_path: Path to a image file.
    :return: PIL.Image object of the car's image and the label(car brand).
    """

    # Load image data.
    image = Image.open(image_path).convert(mode="RGB")

    return image


def load_car_json_data(car_json_path):
    """
    Load car json file.
    :param car_json_path: Path to a car json file.
    :return: PIL.Image object of the car's image and the label(car brand).
    """
    import json
    import base64
    import io
    from PIL import Image

    with open(car_json_path, 'r') as json_data:
        j = json.load(json_data)

        # Name of this image.
        car_name = j["brand"].split("#")[1]

        # Image bytes.
        image_bytes = base64.b64decode(j["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert(mode="RGB")

    return image, car_name

def load_car_color_json_data(car_json_path):
    """
    load car json file.
    ：param car_json_path: Path to a car json file.
    """
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
        else:
            return None, None

    return image, color


# Convert car json to image.
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        the_path = sys.argv[1:]
    else:
        the_path = ["./"]

    for a_path in the_path:
        a_path = a_path.encode("utf-8")
        if os.path.isfile(a_path):
            img, name = load_car_json_data(a_path)
            with open(a_path[:-4] + ".txt", 'w') as label_file:
                label_file.write(name.encode("utf-8"))
            img.save(a_path[:-4] + ".jpg")

        for path, _, files in os.walk(a_path):
            for a_file in files:
                img, name = load_car_json_data(a_file)
                with open(os.path.join(path, a_file[:-4] + ".txt"), 'w') as label_file:
                    label_file.write(name.encode("utf-8"))
                img.save(os.path.join(path, a_file[:-4] + ".jpg"))
