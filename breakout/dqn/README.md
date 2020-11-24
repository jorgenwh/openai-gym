Solving the openai-gym cartpole environment using a deep Q learning network.
Some implementation details were mimicked from [this](https://keras.io/examples/rl/deep_q_network_breakout/) example, such as the network architecture and the update frequency.

Run the script:

    python main.py

The script can take the following arguments:

    --help          : Show help message.
    --render        : Render the game screen.
    --eval          : Load and evaluate a model from the 'models/' folder.
    --save          : Save the model under 'models/[name]' when it solves 
                            the environment. This argument takes a name for the model.
    --plot          : Plot the average rewards for each episode. This argument takes
                            a filename for the plot file.
    --cuda          : Whether to use cuda for the neural network.
    --maxeps        : The maximum amount of games played before termination.