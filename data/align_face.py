import os

from adiencealign.pipeline.CascadeFaceAligner import CascadeFaceAligner


def pipeline(input_path, aligned_path):
    for path, subdirs, files in os.walk(input_path):
        if len(files) < 1:
            continue

        face_path = path.replace(input_path, os.path.join(aligned_path, "face"), 1)
        output_path = path.replace(input_path, os.path.join(aligned_path, "aligned"), 1)
        os.makedirs(face_path, exist_ok=True)

        cascade_face_aligner = CascadeFaceAligner()
        cascade_face_aligner.detect_faces(path, face_path)
        cascade_face_aligner.align_faces(input_images=face_path,
                                         output_path=output_path,
                                         fidu_max_size=200 * 200,
                                         fidu_min_size=50 * 50,
                                         is_align=True,
                                         is_draw_fidu=True,
                                         delete_no_fidu=True)


if __name__ == "__main__":
    import argparse

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="find and align face in source images.")

    parser.add_argument("-i", "--input-path",
                        help="path to input directory")
    parser.add_argument("-o", "--output-path",
                        help="path to output directory")
    args = parser.parse_args()

    pipeline(args.input_path, args.output_path)
