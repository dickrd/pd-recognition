from data.convert import TfWriter, AutoList


def load_car_json_data(car_json_path, dry_run=False, class_filter=None,
                       resize=(512, 512), crop_percentage=1.0):
    """
    Load car json file.
    :param car_json_path: Path to a car json file.
    :param dry_run: Image data will not be processed and be None if set.
    :param class_filter: A list of desired car models. If provided, all other models will be discarded.
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

        if class_filter:
            import re

            model = j["brand"].lower()
            pattern = re.sub(r"\(.+\)", "", model).strip()

            found = False
            for item in class_filter:
                target = re.sub(r"\(.+\)", "", item.lower()).strip()
                if target in pattern:
                    found = True
                    print "Found: " + model.encode("utf-8")
                    break
            if not found:
                return None, None
        if dry_run:
            return None, label

        # Image bytes.
        image_bytes = base64.b64decode(j["image"])

        # Pre-process image.
        image = Image.open(io.BytesIO(image_bytes))

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


def main():
    """
    Tool to convert saved json data to tfrecords.
    When calling from cli, a directory to json files are required.
    It is assumed that all files in the directory and its subdirectories are json file 
    that has field "image" which is base64 encoded image bytes and "brand" which is '#' 
    separated brand category like "car#tesla#tesla model 3#tesla model 3 2018".
    
    The random chance is for example [0.2, 0.3] indicating that there is a 20% chance a
    specific image will be written to the second tfrecord file and if it fails, there will be a 
    30% chance it will be written to the third tfrecord file and if all these fails, it will, 
    default to be written to the first file.
    :return: None 
    """
    import argparse
    import os
    import random
    import json

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="convert image to tfrecords file.")
    parser.add_argument("input",
                        help="path to image files")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="generate json statistics only")
    parser.add_argument("-n", "--name-count",
                        help="path to json statistics file")
    parser.add_argument("-c", "--car-class",
                        help="path to json car class file")
    parser.add_argument("-C", "--desired-class", nargs='+',
                        help="name of desired car class(requires --car-class)")
    parser.add_argument("-l", "--limit", type=int, nargs=2,
                        help="limit data set size of every label to(requires --name-count)")
    parser.add_argument("-o", "--output-path", default="./",
                        help="path to store result tfrecords")
    parser.add_argument("-r", "--random-chance", default=[], type=float, nargs='*',
                        help="chance to split an example to a new tfrecord file")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="size to resize image files to")
    parser.add_argument("--crop", default=1.0, type=float,
                        help="percentage of file after the crop, starting from center")
    args = parser.parse_args()

    class_filter = None
    if args.desired_class:
        if not args.car_class:
            print "Json car class file(--car-class) is required."
            return

        with open(args.car_class, 'r') as car_class_file:
            car_class = json.load(car_class_file)

        class_filter = set()
        for a_class in args.desired_class:
            class_filter.update(car_class[a_class])
        print "Cars will be added: "
        print class_filter

    if args.dry_run:
        print "This run will only summarize data statistics."
        name_count = {}
        # Walk through given path, to find car images.
        for path, subdirs, files in os.walk(args.input):
            print "In: " + path
            for a_file in files:
                # Try to read car image.
                # noinspection PyBroadException
                try:
                    img, name = load_car_json_data(os.path.join(path, a_file), class_filter=class_filter,
                                                   dry_run=True)
                    if not name:
                        continue

                    # Get statistics.
                    if name not in name_count:
                        name_count[name] = 1
                    else:
                        name_count[name] += 1
                except Exception as e:
                    print "Error reading " + a_file + ": " + repr(e)
        print "All images processed."

        # Write results.
        with open(os.path.join(args.output_path, "name_count.json"), 'w') as count_file:
            count_file.write(json.dumps(name_count))

        # Print statistics.
        sorted_kv = sorted(name_count.items(), key=lambda x: x[1])
        for item in sorted_kv:
            print "({0}, {1})\t".format(item[0].encode("utf-8"), item[1]),
        print "\n",

        # Print summary.
        mid_kv = sorted_kv[len(sorted_kv)/2]
        small_kv = sorted_kv[0]
        big_kv = sorted_kv[-1]
        print "Middle number: ({0}, {1})".format(mid_kv[0].encode("utf-8"), mid_kv[1])
        print "Largest label: ({0}, {1})".format(big_kv[0].encode("utf-8"), big_kv[1])
        print "Smallest label: ({0}, {1})".format(small_kv[0].encode("utf-8"), small_kv[1])
        return

    if args.limit:
        if not args.name_count:
            print "Json statistics file(--name-count) is required."
            return
        with open(args.name_count, 'r') as name_count_file:
            name_count = json.load(name_count_file)
        print "Image count limit: {0}~{1}".format(args.limit[0], args.limit[1])
    else:
        name_count = None

    print "Images will cropped to: x" + str(args.crop)
    print "And resized to: " + str(args.resize) + "x" + str(args.resize)
    # File path list of tfrecord files.
    result_files = [os.path.join(args.output_path, "train.tfrecords")]
    # Writers list.
    writers = [TfWriter(result_files[0])]
    # Example count of each tfrecord file.
    count = [0]
    name_wrote_count = AutoList()

    print "Default output file: " + result_files[0]
    for index, item in enumerate(args.random_chance):
        result_files.append(os.path.join(args.output_path, "test_" + str(index) + ".tfrecords"))
        writers.append(TfWriter(result_files[index + 1]))
        count.append(0)
        print "Chance for example to file " + result_files[index + 1] + ": " + str(item)

    name_labels = {}
    # Walk through given path, to find car images.
    for path, subdirs, files in os.walk(args.input):
        print "In: " + path
        for a_file in files:
            # Try to read car image.
            # noinspection PyBroadException
            try:
                img, name = load_car_json_data(os.path.join(path, a_file), class_filter=class_filter,
                                               resize=(args.resize, args.resize),
                                               crop_percentage=args.crop)
                if not name:
                    continue

                # Skip unsatisfied files.
                if name_count and name_count[name] < args.limit[0]:
                    continue

                # Convert car brand name to int. Save the conversion for converting back.
                if name not in name_labels:
                    label = len(name_labels)
                    name_labels[name] = label
                else:
                    label = name_labels[name]

                # Skip extra images.
                if args.limit and name_wrote_count[label] > args.limit[1]:
                    continue

                # If this car image has been write to a tfrecord file.
                write = False

                # Determine which tfrecord file to write this image to.
                for index, item in enumerate(args.random_chance):
                    if random.random() < item:
                        writers[index + 1].write(img, label)
                        count[index + 1] += 1
                        write = True
                        break
                # Default write to this file.
                if not write:
                    writers[0].write(img, label)
                    count[0] += 1

                name_wrote_count[label] += 1
            except Exception as e:
                print "Error reading " + a_file + ": " + repr(e)

        # Count example for every directory read.
        for index, file_name in enumerate(result_files):
            print "Current examples in " + file_name + ": " + str(count[index])

    print "All images processed."

    # Close the writers.
    for writer in writers:
        writer.close()

    print "Labels wrote: {0}".format(name_wrote_count)
    # Save name label conversion.
    with open(os.path.join(args.output_path, "name_labels.json"), 'w') as label_file:
        label_file.write(json.dumps(name_labels))
    print "Done."


if __name__ == "__main__":
    main()
