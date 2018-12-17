# Very Deep Convolutional Networks for Large Scale Image Recognition
* Convolution Stride = 1
* All Filter = 3 x 3
* 5 max pooling rather than spartial pooling
* Architecture:
    1. conv3 - 64 (relu)
    2. conv3 - 64 (relu)
    3. max_pool
    4. conv3 - 128 (relu)
    5. conv3 - 128 (relu)
    6. max_pool
    7. conv3 - 256 (relu)
    8. conv3 - 256 (relu)
    9. conv3 - 256 (relu)
    10. max_pool
    11. conv3 - 512 (relu)
    12. conv3 - 512 (relu)
    13. conv3 - 512 (relu)
    14. max_pool
    15. conv3 - 512 (relu)
    16. conv3 - 512 (relu)
    17. conv3 - 512 (relu)
    18. max_pool
    19. FC
## VGG19: 
#
    network = input_data(shape=[None,IMG_SIZE,IMG_SIZE,3])

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 10, activation='softmax')

    network = regression(network, optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)

    # Training
    model = tflearn.DNN(network, checkpoint_path='model_vgg',
                        max_checkpoints=1, tensorboard_verbose=0)
