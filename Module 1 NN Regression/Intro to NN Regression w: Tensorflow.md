What is a regression problem?

    Example of regression problems

    -> How much will this house sell for?
    -> How many ppl will buy this app?
    -> How much will my health insurance be?
    -> How much should i save each week for fuel?

    The keypoint of a regression problem is predicting a number

    There are other types of problems where you can turn them into a regression problem
    This can be predicting the coordinates of a certain object in an image (e.g. face detection)

    What we're going to cover (broadly)
    -> Architecture of a nn regression model
    -> Input shapes and output shapes of a regression model (features and labels, respectively)
    -> Creating custom data to view and fit
    -> Steps in modelling
        -> Creating a model, compiling a model, fitting, and evaluating
    -> Different evaluation methods
    -> Saving and loading models



Inputs and Outputs of a Regression Model

    Let's say we want to find the price of a house
    We have to grab all the possible data we can get abt the house
    From the number of bedrooms, bathrooms, garages, etc.

    Regression analysis is a set of statistical processes for estimating
    the RELATIONSHIP btwn a DEPENDENT VARIABLE (labels/outcome variable/output) and one
    or more INDEPENDENT VARIABLE (features/input)

    For this example, we'll say that the features/independent variables are 4 bedrooms, 
    2 bathrooms, and 2 garages

    For the ML Algorithm, we're going to have to numerically encode the features
    We can one-hot encode (1 and 0)

                        # of whatever
                        1   2   3   4
                        [[0, 0, 0, 1],          Bedroom
                        [0, 1, 0, 0],           Bathroom
                        [0, 1, 0, 0]]           Garage

    (Often, the machine learning algorithm already exists; if not, make one)

    For this scenario, the predicted output for the house's sale price is 939,700
    while the actual output is 940,000. Using supervised ML, we can get a more accurate
    and precise predicted output by fine tuning the output every time                      

    Input and Output Shapes:
        With all things said, what might be the shape of our inputs for our example?
        Shape = [3]; this is bc we got 3 input vectors

        What abt the shape of output?
        Shape = [1]; this is bc we get only 1 output vector which is the price of the house
        


Architecture of a NN Regression Model

    Typically, we would have the following as the hyperparameter(changeable)
        Input layer shape           (Same shape as # of features (e.g. 3 for # bedrooms, # bathrooms, # garages))
        Hidden layers               (Problem specific; at least one layer)
        Neurons per hidden layer    (Problem specific; usually btwn 10 and 100)
        Output layer shape          (Same shape as desired prediction shape (e.g. 1 for housing price))
        Hidden activation           (Usually ReLU (rectified linear unit))
        Output activation           (None, ReLU, logistic/tanh)
        Loss function               (MSE (mean square error) or MAE (mean absolute error)/Huber (combinatinon of MSE/MAE) if outliers)
        Optimizer                   (SGD (stochastic gradient descent), Adam)

    Creating a model:

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(3, )),                    __          INPUT LAYER SHAPE   ___ (The '100': NEURONS)
            tf.keras.layers.Dense(100, activation='relu')     |                             |
            tf.keras.layers.Dense(100, activation='relu')     |----     HIDDEN LAYERS    ---|
            tf.keras.layers.Dense(100, activation='relu')   __|                             |__ (The 'activation': HIDDEN ACTIVATION)
            tf.keras.layers.Dense(1, activation=None)                   (The '1': OUTPUT LAYER SHAPE) (The 'activation': OUTPUT ACTIVATION)
        ])
    
    Compile the model:
        
        model.compile(loss=tf.keras.losses.mae,                         LOSS FUNCTION (This will measure how wrong the NN relationships are)
                    optimizer=tf.keras.optimizers.Adam(lr=0.001),       OPTIMIZER (Inform the NN how it should improve the patterns to reduce the loss function)
                    metrics=["mae"])
    
    Fit the model

        model.fit(x_train, y_train, epochs=100)                         