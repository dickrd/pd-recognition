import os

from adiencealign.pipeline.CascadeFaceAligner import CascadeFaceAligner


def pipeline(input_path, aligned_path):
    for path, subdirs, files in os.walk(input_path):
        if len(files) < 1:
            continue

        face_path = path.replace(input_path, os.path.join(aligned_path, "face/"), 1)
        output_path = path.replace(input_path, os.path.join(aligned_path, "aligned/"), 1)
        try:
            os.makedirs(face_path)
            os.makedirs(output_path)
        except OSError:
            pass

        cascade_face_aligner = CascadeFaceAligner()
        cascade_face_aligner.detect_faces(path, face_path)
        cascade_face_aligner.align_faces(input_images=face_path,
                                         output_path=output_path,
                                         fidu_max_size=200 * 200,
                                         fidu_min_size=50 * 50,
                                         is_align=True,
                                         is_draw_fidu=False,
                                         delete_no_fidu=True)


def process_img(img_path):
    output_path = "aligns/"
    try:
        os.makedirs(output_path)
    except OSError:
        pass

    cascade_face_aligner = CascadeFaceAligner()
    faces_file = cascade_face_aligner.face_finder.create_faces_file(img_path, is_overwrite=False, target_file='aligns/faces.txt')
    sub_images_files = cascade_face_aligner.face_finder.create_sub_images_from_file(original_image_file=img_path,
                                                                                    faces_file=faces_file,
                                                                                    target_folder=output_path,
                                                                                    img_type='jpg')
    cascade_face_aligner.align_faces(input_images=sub_images_files,
                                     output_path=output_path,
                                     fidu_max_size=200 * 200,
                                     fidu_min_size=50 * 50,
                                     is_align=True,
                                     is_draw_fidu=True,
                                     delete_no_fidu=True)

    result_img_path = []
    for img_file in sub_images_files:
        _, img_name = os.path.split(img_file)
        result_img_path.append(os.path.join(output_path, img_name.rsplit('.',1)[0] + '.aligned.png'))
    return result_img_path


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
