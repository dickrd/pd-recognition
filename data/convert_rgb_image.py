import os

from data.common import load_image_data, generate_name, AutoList, TfWriter


def main():
    """
    Tool to convert rgb images to tfrecords.
    
    The random chance is for example [0.2, 0.3] indicating that there is a 20% chance a
    specific image will be written to the second tfrecord file and if it fails, there will be a 
    30% chance it will be written to the third tfrecord file and if all these fails, it will, 
    default to be written to the first file.
    :return: None 
    """
    import argparse
    import random
    import json

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="convert image to tfrecords file.")

    parser.add_argument("-i", "--input-path", nargs='+',
                        help="path to image files")
    parser.add_argument("-l", "--limit", type=int,
                        help="limit data set size of every label to a fixed number")
    parser.add_argument("-t", "--label-index", type=int, default=0,
                        help="which directory in path will be label, count backwards")
    parser.add_argument("-o", "--output-path", default="./",
                        help="path to store result tfrecords")
    parser.add_argument("-r", "--random-chance", default=[], type=float, nargs='*',
                        help="chance to split an example to a new tfrecord file")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="size to resize image files to")
    args = parser.parse_args()

    if not args.input_path:
        print "Must specify image paths(--image-path)!"
        return

    print "Image will be resized to: " + str(args.resize) + "x" + str(args.resize)

    # Vars for write images to different files and count.
    # File path list of tfrecord files.
    result_files = [os.path.join(args.output_path, "train.tfrecords")]
    # Writers list.
    writers = [TfWriter(result_files[0])]
    # Count of each tfrecord file.
    count = [0]
    # Count for each name.
    name_wrote_count = AutoList()

    print "Default output file: " + result_files[0]
    for index, item in enumerate(args.random_chance):
        result_files.append(os.path.join(args.output_path, "test_" + str(index) + ".tfrecords"))
        writers.append(TfWriter(result_files[index + 1]))
        count.append(0)
        print "Chance for example to file " + result_files[index + 1] + ": " + str(item)

    # Name to label conversion.
    name_labels = {}
    for directory in args.input_path:
        # Walk through given path, to find car images.
        for path, subdirs, files in os.walk(directory):
            print "In: " + path
            for a_file in files:
                # Try to read image.
                # noinspection PyBroadException
                try:
                    current_file_path = os.path.join(path, a_file)

                    img = load_image_data(current_file_path,
                                          resize=(args.resize, args.resize))
                    name = generate_name(current_file_path, index=args.label_index)

                    if not name:
                        continue

                    # Convert readable name to int label. Save the conversion for converting back.
                    if name not in name_labels:
                        label = len(name_labels)
                        name_labels[name] = label
                    else:
                        label = name_labels[name]

                    # Skip extra images.
                    if args.limit and name_wrote_count[label] > args.limit:
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
