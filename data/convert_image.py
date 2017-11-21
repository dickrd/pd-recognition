import json
import os

from data.common import AutoList
from data.image_loading import load_compcar_with_crop, load_image_data, load_car_json_data
from data.name_generation import generate_name_from_path


def convert_image(input_path, label_index, resize,
                  limit, name_count, random_chance,
                  output_path, dry_run,
                  load_image, packed, crop_percentage, name_filter):
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
    writers = [TfWriter(result_files[0])]
    # Count of each tfrecord file.
    file_wrote_count = [0]
    # Count for each name.
    name_wrote_count = AutoList()

    print "Default output file: " + result_files[0]
    for index, item in enumerate(random_chance):
        result_files.append(os.path.join(output_path, "test_" + str(index) + ".tfrecords"))
        writers.append(TfWriter(result_files[index + 1]))
        file_wrote_count.append(0)
        print "Chance for example to file " + result_files[index + 1] + ": " + str(item)

    # Name to label conversion.
    name_labels = {}
    for directory in input_path:
        # Walk through given path, to find car images.
        for path, subdirs, files in os.walk(directory):
            print "In: " + path
            for a_file in files:
                # Try to read image.
                # noinspection PyBroadException
                try:
                    current_file_path = os.path.join(path, a_file)

                    if packed:
                        img, name = load_image(current_file_path, resize=(resize, resize),
                                               crop_percentage=crop_percentage, name_filter=name_filter)
                    else:
                        img = load_image(current_file_path, resize=(resize, resize))
                        name = generate_name_from_path(current_file_path, index=label_index)

                    # Skip unqualified files.
                    if not name:
                        continue
                    if name_count and name_count[name] < limit:
                        continue

                    # Convert readable name to int label. Save the conversion for converting back.
                    if name not in name_labels:
                        label = len(name_labels)
                        name_labels[name] = label
                    else:
                        label = name_labels[name]

                    # Skip extra images.
                    if limit and name_wrote_count[label] > limit:
                        continue

                    # If this car image has been write to a tfrecord file.
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

                    name_wrote_count[label] += 1
                except Exception as e:
                    print "Error reading " + a_file + ": " + repr(e)

            # Count example for every directory read.
            for index, file_name in enumerate(result_files):
                print "Current examples in " + file_name + ": " + str(file_wrote_count[index])

    print "All images processed."

    # Close the writers.
    for writer in writers:
        writer.close()
    # Save name label conversion.
    with open(os.path.join(output_path, "name_labels.json"), 'w') as label_file:
        label_file.write(json.dumps(name_labels))

    return name_wrote_count


def print_dataset_summary(name_wrote_count, write_path=None):
    if write_path:
        # Write results.
        with open(os.path.join(write_path, "name_count.json"), 'w') as count_file:
            count_file.write(json.dumps(name_wrote_count))

    # Print statistics.
    sorted_kv = sorted(name_wrote_count.items(), key=lambda x: x[1])
    for item in sorted_kv:
        print "({0}, {1})\t".format(item[0].encode("utf-8"), item[1]),
    print "\n",

    # Print summary.
    mid_kv = sorted_kv[len(sorted_kv)/2]
    small_kv = sorted_kv[0]
    big_kv = sorted_kv[-1]
    print "Middle:\t({0}, {1})".format(mid_kv[0].encode("utf-8"), mid_kv[1])
    print "Largest:\t({0}, {1})".format(big_kv[0].encode("utf-8"), big_kv[1])
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
                        help="path to image files")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="generate json statistics only")
    parser.add_argument("-p", "--pre-process", default="general",
                        help="which process to take (general, jsoncar, compcar)")
    parser.add_argument("-l", "--label-index", type=int, default=-2,
                        help="which directory in path will be label")
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
    args = parser.parse_args()

    if not args.input_path:
        print "Must specify image paths(--image-path)!"
        return

    packed = False
    if args.pre_process == "general":
        load_image = load_image_data
    elif args.pre_process == "compcar":
        import re
        label_path = re.sub(r"/image/?", r"/label/", args.input_path)

        if os.path.isdir(label_path):
            print "Crop compcar full images using label: " + label_path
            load_image = load_compcar_with_crop
        else:
            print "Error reading label directory: " + label_path
            return
    elif args.pre_process == "jsoncar":
        load_image = load_car_json_data
        packed = True
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

    name_count = None
    if args.limit:
        print "Image in each name will be limited to {0}".format(args.limit),
        if args.name_count:
            with open(args.name_count, 'r') as name_count_file:
                name_count = json.load(name_count_file)
                print " and unqualified names will be skipped."
        else:
            print "."

    print "Image will be resized to: " + str(args.resize) + "x" + str(args.resize)

    name_wrote_count = convert_image(input_path=args.input_path, label_index=args.label_index, resize=args.resize,
                                limit=args.limit, name_count=name_count, random_chance=args.random_chance,
                                output_path=args.output_path, dry_run=args.dry_run,
                                load_image=load_image, packed=packed, crop_percentage=args.crop, name_filter=name_filter)

    print_dataset_summary(dict(name_wrote_count))
    print "Done."


if __name__ == "__main__":
    _main()
