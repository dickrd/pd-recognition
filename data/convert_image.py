#coding:utf-8
import json
import os

from data.image_loading import load_compcar_with_crop, load_image_data, load_car_json_data, get_label_path_of_compcar, \
    AdienceUtil, normalize_image
from data.image_loading import load_car_color_json_data
from data.name_generation import generate_name_from_path


def extract_image(input_path, resize, limit, output_path,
                  regression,
                  report_rate=50):
    from data.common import TfReader
    from PIL import Image
    import numpy as np
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    reader = TfReader(data_path=input_path, regression=regression, size=(resize, resize))
    img_op, label_op = reader.read(batch_size=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run([init_op])
        tf.train.start_queue_runners(sess)
        step_count = 0
        label_wrote_count = {}
        try:
            while True:
                step_count += 1
                img, label = sess.run([img_op, label_op])
                img = img[0]
                label = label[0]

                output_dir = os.path.join(output_path, str(label))

                # Count name wrote times.
                if label not in label_wrote_count:
                    label_wrote_count[label] = 0
                    if not os.path.isdir(output_dir):
                        os.mkdir(output_dir)
                # Skip extra images.
                elif limit and label_wrote_count[label] >= limit:
                    continue

                Image.fromarray(np.uint8(img), "RGB").save(os.path.join(output_dir,
                                                                        "step_{0}.jpg".format(step_count)))
                label_wrote_count[label] += 1
                if step_count % report_rate == 0:
                    print "{0} steps passed with label wrote: ".format(step_count),
                    print label_wrote_count
        except tf.errors.OutOfRangeError:
            print "All images used in {0} steps.".format(step_count)


def convert_image(input_path, label_index, resize,
                  limit, name_count, random_chance,
                  output_path, dry_run, regression,
                  load_image, packed, crop_percentage, name_filter,
                  walk=os.walk, generate_name=generate_name_from_path, name_label_path=None):
    import random

    if dry_run:
        from data.common import DummyWriter as TfWriter
        print "Using dummy writer that writes nothing."
    else:
        from data.common import TfWriter

    # Vars for write images to different files and count.
    # File path list of tfrecord files.
    result_files = [os.path.join(output_path, "train.tfrecords")]
    # Writers list.
    writers = [TfWriter(result_files[0], regression=regression)]
    # Count of each tfrecord file.
    file_wrote_count = [0]
    # Count for each name.
    name_wrote_count = {}

    print "Default output file: " + result_files[0]
    for index, item in enumerate(random_chance):
        result_files.append(os.path.join(output_path, "test_" + str(index) + ".tfrecords"))
        writers.append(TfWriter(result_files[index + 1], regression=regression))
        file_wrote_count.append(0)
        print "Chance for example to file " + result_files[index + 1] + ": " + str(item)

    # Name to label conversion.
    if name_label_path:
        with open(name_label_path, 'r') as name_label_file:
            name_labels = json.loads(name_label_file)
    else:
        name_labels = {}
    img = None
    last_path= None
    for directory in input_path:
        # Walk through given path, to find car images.
        for path, _, files in walk(directory):
            for a_file in files:
                # Try to read image.
                # noinspection PyBroadException
                try:
                    current_file_path = os.path.join(path, a_file)

                    if packed:
                        img, name = load_image(current_file_path)
                    else:
                        img = load_image(current_file_path)
                        name = generate_name(current_file_path, index=label_index)

                    if name_filter and name:
                        # Search in name filter.
                        found = False
                        for approved_name in name_filter:
                            # If the name of this image is approved.
                            if approved_name == name:
                                found = True
                                break
                        if not found:
                            name = None

                    # Skip unqualified files.
                    if not name:
                        continue
                    if name_count and name_count[name] < limit:
                        continue

                    if regression:
                        # Convert label to float.
                        label = float(name)
                    else:
                        # Convert readable name to int label. Save the conversion for converting back.
                        if name not in name_labels:
                            label = len(name_labels)
                            name_labels[name] = label
                        else:
                            label = name_labels[name]

                    # Count name wrote times.
                    if name not in name_wrote_count:
                        name_wrote_count[name] = 0
                    # Skip extra images.
                    elif limit and name_wrote_count[name] >= limit:
                        continue

                    # Resize and crop if necessary.
                    img = normalize_image(img, resize=(resize, resize), crop_percentage=crop_percentage)
                    # This image has not been write to a tfrecord file.
                    write = False

                    # Determine which tfrecord file to write this image to.
                    for index, item in enumerate(random_chance):
                        if random.random() < item:
                            writers[index + 1].write(img, label)
                            file_wrote_count[index + 1] += 1
                            write = True
                            break
                    # Default write to this file.
                    if not write:
                        writers[0].write(img, label)
                        file_wrote_count[0] += 1

                    name_wrote_count[name] += 1
                except Exception as e:
                    print "Error reading " + a_file + ": " + repr(e)


            if not last_path:
                last_path = path
            if last_path != path:
                # Print directory summary.
                print "{0} has been processed.".format(last_path)
                for index, file_name in enumerate(result_files):
                    print "Current examples in " + file_name + ": " + str(file_wrote_count[index])
                last_path = path

        # Print directory summary.
        print "{0} has been processed.".format(last_path)
        for index, file_name in enumerate(result_files):
            print "Current examples in " + file_name + ": " + str(file_wrote_count[index])

    if img:
        sample_path = os.path.join(output_path, "sample.jpg")
        img.save(sample_path)
        print "Sample image in " + sample_path

    print "All images processed."

    # Close the writers.
    for writer in writers:
        writer.close()

    if not regression:
        # Save name label conversion.
        label_path = os.path.join(output_path, "name_labels.json")
        with open(label_path, 'w') as label_file:
            label_file.write(json.dumps(name_labels))
        print "Saved name label conversion: " + label_path

    return name_wrote_count


def print_dataset_summary(name_wrote_count=None, store_path=None, read_mode=False):
    if store_path:
        file_path = os.path.join(store_path, "name_count.json")
        if read_mode:
            # Read result.
            with open(file_path, 'r') as count_file:
                name_wrote_count = json.loads(count_file)
        elif name_wrote_count:
            # Write results.
            with open(file_path, 'w') as count_file:
                count_file.write(json.dumps(name_wrote_count))

    # Print statistics.
    print "Statics:"

    if not name_wrote_count or len(name_wrote_count) == 0:
        print "\tNone"
        return

    sorted_kv = sorted(name_wrote_count.items(), key=lambda x: x[1])
    count = 0
    for item in sorted_kv:
        print "\t<{0}, {1}>".format(item[0].encode("utf-8"), item[1]),
        count += 1
        if count % 5 == 0:
            print ""
    print ""

    # Print summary.
    mid_kv = sorted_kv[len(sorted_kv)/2]
    small_kv = sorted_kv[0]
    big_kv = sorted_kv[-1]
    print "Middle:  \t({0}, {1})".format(mid_kv[0].encode("utf-8"), mid_kv[1])
    print "Largest: \t({0}, {1})".format(big_kv[0].encode("utf-8"), big_kv[1])
    print "Smallest:\t({0}, {1})".format(small_kv[0].encode("utf-8"), small_kv[1])


def _main():
    """
    Tool to convert rgb images to tfrecords.

    The random chance is for example [0.2, 0.3] indicating that there is a 20% chance a
    specific image will be written to the second tfrecord file and if it fails, there will be a
    30% chance it will be written to the third tfrecord file and if all these fails, it will,
    default to be written to the first file.

    Filter json file is a dict from different categories (string) to names list.
    :return: None
    """
    import argparse

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="convert image to tfrecords file.")

    parser.add_argument("-i", "--input-path", nargs='+',
                        help="path to input files")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="generate json statistics only")
    parser.add_argument("-p", "--pre-process", default="general",
                        help="which process to take (general, jsoncar, compcar, carcolor, adience)")
    parser.add_argument("-l", "--label-index", type=int, default=-2,
                        help="which directory in path will be label")
    parser.add_argument("-n", "--name-label",
                        help="path to json file that contains a map of readable name to class label.")
    parser.add_argument("-o", "--output-path", default="./",
                        help="path to store result tfrecords")
    parser.add_argument("-r", "--random-chance", default=[], type=float, nargs='*',
                        help="chance to split an example to a new tfrecord file")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="size to resize image files to")

    parser.add_argument("--crop", default=1.0, type=float,
                        help="percentage of file after the crop, starting from center")
    parser.add_argument("--name-filter", default=[], nargs='+',
                        help="path to filter json file followed by desired categories")
    parser.add_argument("--limit", type=int,
                        help="limit data set size of every label to a fixed number")
    parser.add_argument("--name-count",
                        help="path to dry run generated json file for limit to skip unqualified names")
    parser.add_argument("--regression", action="store_true",
                        help="set to make labels as float numbers parsed from names")
    parser.add_argument("--reverse", action="store_true",
                        help="read tfrecord file and extract images in it")
    parser.add_argument("--print-statistics", action="store_true",
                        help="print statistics of a converted dataset")
    args = parser.parse_args()

    if not args.input_path:
        print "Must specify input paths(--input-path)!"
        return

    if args.print_statistics:
        print_dataset_summary(store_path=args.input_path, read_mode=True)

    if args.regression:
        print "Regression set."
    elif args.name_label:
        print "Reuse name label map file: " + args.name_label

    name_count = None
    if args.limit:
        print "Image in each name will be limited to {0}".format(args.limit),
        if args.name_count:
            with open(args.name_count, 'r') as name_count_file:
                name_count = json.load(name_count_file)
                print " and unqualified names will be skipped."
        else:
            print "."

    print "{0}% of original image will be used.".format(args.crop * 100)
    print "And resized to " + str(args.resize) + "x" + str(args.resize)

    if args.reverse:
        print "Start extract images from tfrecord files."
        extract_image(input_path=args.input_path, resize=args.resize, limit=args.limit, output_path=args.output_path,
                      regression=args.regression)
        return

    walk = os.walk
    packed = False
    generate_name = generate_name_from_path
    if args.pre_process == "general":
        load_image = load_image_data
        print "General pre-process."
    elif args.pre_process == "compcar":
        label_path = []
        for a_input_path in args.input_path:
            a_label_path = get_label_path_of_compcar(a_input_path)
            if os.path.isdir(a_label_path):
                label_path.append(a_label_path)
            else:
                print "No label directories in: " + a_label_path
                return

        print "Compcar pre-process."
        print "Crop compcar full images using labels in: ",
        print label_path
        load_image = load_compcar_with_crop
    elif args.pre_process == "jsoncar":
        load_image = load_car_json_data
        packed = True
        print "Read image in json car file with brand as label."
    elif args.pre_process == "carcolor":
        load_image = load_car_color_json_data
        packed = True
        print "Read image in json car file with color as label."
    elif args.pre_process == "adience":
        adience = AdienceUtil()
        load_image = load_image_data
        walk = adience.walk
        generate_name = adience.name
        print "Adience pre-process."
    else:
        print "Unsupported process type: " + args.pre_process
        return

    name_filter = None
    if args.name_filter:
        if len(args.name_filter) > 1:
            with open(args.name_filter[0], 'r') as filter_category_file:
                filter_category = json.load(filter_category_file)

            name_filter = set()
            for a_name in args.name_filter[1:]:
                name_filter.update(filter_category[a_name])
            print "Names that will be added: ",
            print name_filter
        else:
            print "Unexpected name filter syntax: " + args.name_filter
    else:
        print "Adding all names."

    name_wrote_count = convert_image(input_path=args.input_path, label_index=args.label_index, resize=args.resize,
                                     limit=args.limit, name_count=name_count, random_chance=args.random_chance,
                                     output_path=args.output_path, dry_run=args.dry_run, regression=args.regression,
                                     load_image=load_image, packed=packed, crop_percentage=args.crop, name_filter=name_filter,
                                     walk=walk, generate_name=generate_name, name_label_path=args.name_label)

    print_dataset_summary(name_wrote_count, store_path=args.output_path)
    print "Done."


if __name__ == "__main__":
    _main()
