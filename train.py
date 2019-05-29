import argparse
from keras.preprocessing.image import ImageDataGenerator
from source.model_factory import create_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new model using supplied data.')
    #parser.add_argument('path', type=str, help='Path to folder of images to train on.')
    #parser.add_argument("--width", type=int, help="The width of the images to train the model for.")
    #parser.add_argument("--height", type=int, help="The height of the images to train the model for.")
    args = parser.parse_args()
    #path = args.path

    model = create_model(4000,3000)
    gen = ImageDataGenerator(rotation_range=5, validation_split=0.1, horizontal_flip=True)
    x, y = gen.flow_from_directory("data/", target_size=(4000,3000))
    y = x
    print(x)
    #model.fit_generator(gen, epochs = 5, use_multiprocessing=True)