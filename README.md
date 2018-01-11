pd-recognition
====

Deep pipeline learning in tensorflow.

Dependencies
----

* tensorflow >= 1.2
* numpy
* pillow
* scipy

Usage
----

Convert image to tfrecords:

    python data/convert_image.py [options]

Using CNN model:

    python model/cnn.py <train|auto|test|predict> [options]
