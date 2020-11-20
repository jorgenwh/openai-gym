Solving the openai cartpole environment using a deep Q learning network.

Run the script:

    python main.py

The script can take the following arguments:

    -h, --help          : Show help message.
    -r, --render        : Render the game screen.
    -ev, --eval         : Load and evaluate a model from the 'models/' folder.
    -s, --save          : Save the model under 'models/[name]' when it solves 
                            the environment. This argument takes a name for the model.
    -pl, --plot         : Plot the average rewards for each episode. This argument takes
                            a filename for the plot file.
    -cu, --cuda         : Whether to use cuda for the neural network.
    -max, --maxgames    : The maximum amount of games played before termination.