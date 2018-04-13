from model.cnn_transfer import train_transfer


def _main():
    ####            ####             ####
    #### Start of parameters section ####
    ####            ####             ####
    data_path = [""]
    regression = False
    name_label = ""
    trainable_layers = ["fc6", "fc7"]

    save_path = "model/"
    #### End                         ####

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if regression:
        class_count = 1
        print "Regression set."
    else:
        if not name_label:
            print "Must specify name label file(--name-label)!"
            return

        import json
        with open(name_label, 'r') as label_file:
            label_names = json.load(label_file)
            class_count = len(label_names)

    train_transfer(model_path=save_path, train_data_path=data_path, class_count=class_count,
                   regression=regression, trainable_layers=trainable_layers)

if __name__ == "__main__":
    _main()
