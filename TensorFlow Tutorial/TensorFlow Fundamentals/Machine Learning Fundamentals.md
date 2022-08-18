Machine Learning Fundamentals
    What is Artificial Intelligence?
        The effort to automate intellectual tasks normally performed by humans

    Chess or TicTacToe AI or Pacman etc
        Predefined rules
        No deep learning

    AI can be simple or complex

    AI Bubble:
        Machine Learning figures out the rules for us
            Instead of us hard coding the rules, we would give the data and create the rules that works with the data

                Classical Programming:                              Machine Learning Programming:
                Data    ------------                                Data    ------------                                    
                                    |------ Answers                                     |------ Rules   
                Rules   ------------                                Answers ------------

            Goal for Machine Learning is to get the highest accuracy possible

        Neural Networks: a form of machine learning that uses a layered representation of data 
            Multiple layers of "Machine Learning"
            Inspired by brains but not modeled after the brain

    Features and Labels:
        Features: pretty much the input info (The information given to us)
        Label: the output info (The information that we want the machine to predict)

        Essentially, we give the computer features and let it figure out the rules; it can verify with the labels.
        The machine would then give us "training data" (All features and labels) as it trains. Eventually, we would take away the labels and see if it can give us
        a proper answer with just the features/input data
    
    Different types of Machine Learning:
        Unsupervised, Supervised, and Reinforcement Learning

        Supervised: 
            Have some features that would correspond with labels
            Step 1) Features ---> Labels (Find out what "--->" is )
            Step 2) Features ---> ??? (After finding out what "--->" is, find the labels (Output Data) without help using those rules)
            Pretty much, it makes a prediction, fine tune the prediction with "Labels" we have, then do it again

        Unsupervised:
            When we only have features and we want the machine to figure out what the labels are

        Reinforcement:
            No data at all; you have an agent, environment, and reward
            Think abt the flappy bird ai 
            It would randomly explore and tries to maximize its reward



Introduction to Tensorflow:
    Refer to this: https://colab.research.google.com/drive/1F_EWVKa8rbMXi3_fG0w7AtcscFq7Hi7B#forceEdit=true&sandboxMode=true

    Some things we can do with TensorFlow:
        Image classification
        Data cluster
        Regression
        Reinforcement Learning
        Natural Language Processing
    
    Its a library of tools that would help omit the use of heavy math

    TensorFlow works with sessions and graphs:
        Graphs: Nothing is really computed or stored but defined
        Session: Allows parts of the graph to be executed
    
    Tensors:
        A generalization of vectors and matrices to potentially higher dimensions
        Pretty much the inputs and outputs in numerical encoding 

Things to Accomplish in the Course:
    Tensorflow basics and fundamentals
    Preprocessing data (Getting it into tensors)
    Building and using pretrained deep learning models
    Fitting a model to the data (Learning patterns)
    Making predictions with a model (using patterns)
    Evaluating model predictions
    Saving and loading models
    Using a trained model to make predictions on custom data

    The concept: A chef vs a chemist
        Machine learning is like a chef
    
    The "TensorFlow":
    1) Get data ready (Turn it into tensors/numbers)
    2) Build/pick a pretrained model
    3) Fit the model to the data and make a prediction
    4) Evaluate the prediction 
    5) Improve thru experiments
    6) Save and Load






            

