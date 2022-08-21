NN Classification w/ Tensorflow

What is a classification problem?

Types of Classification Problems:

    Binary classification:
        Is it one thing or another?
        (e.g.) is it email spam or not spam? To be or not to be

    Multiclass Classification:
        Same but more than one thing or another
        (e.g.) Is this a photo of sushi, steak or pizza?
        multiple objects, one tag per

    Multilabel Classification:
        Multiple label options per sample
        (e.g.) 
        "What tags should this one article have?" *Shows article of Deep Learning* 
        Possible tags: Machine learning; Representation learning; Artificial Intelligence
        One object, multiple tags per



Classification inputs and outputs

    Get as much information you can get from the object/picture (e.g. color, dimensions, etc.)

    Numerically encode the data (normalization commonly used)

    Incorporate a ML Algorithm

    Get Predicted Output

    Turn Predicted output to actual output



Classification Input and Output Tensor Shapes

    (For an image classification example)


Typical Architecture of a Classification Model

    Create a model
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape = (224, 224, 3)),
        tf.keras.layers.Dense(100, activation = "relu")
        tf.keras.layers.Dense(3, activation = "softmax")
    ])

    Compile 

    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics = ["accuracy"]
    )

    Fit

    model.fit(X_train, y_train, epochs = 5)

    Evaluate 

    model.evaluate(X_test, y_test)

    (Note: Multiclass and Multilabel Classification have similar code)
    