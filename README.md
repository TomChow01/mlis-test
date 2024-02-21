# Internship Technical Challenge

## Details

Please modify and commit this file to fill in your details below:

Full Name: 
Student ID:
Username: 


## Task

Your task is to finish implementing the functions provided in `interview/custom_slm.py`, using the docstrings and unit tests as a specification for what the function does. This class implements a custom language model with a restricted vocabulary, provided in the `data/1000.txt` file. All tokens generated using this vocabulary list should be able to be generated by this custom class.

Please complete an implementation for all function bodies that are currently filled with "`pass`".

**Do not** modify function names or variable names of the class as these will be used for testing. Some functions have already been implemented for you to save time. These implementations do not need to be modified to pass. Inside the `data` folder, we have provided a file that can be used to construct your model object for testing.

Make sure you can pass the provided unit tests and verify your implementation is correct. **Please note** that passing the unit tests is **not** sufficient to guarantee a correct implementation. 

Once you are satisfied with your submission, make sure to commit your final version to the main branch in this repository.

## Getting Started

To start, you must locally clone this repository to your local machine using Git. Use Git best practices while implementing your solution. Make sure to test your implementation using the data provided and to verify yourself that your code is correct and robust.

## Rules

- **DO NOT** use any AI assistance tools in your submission, such as ChatGPT or GitHub Copilot. If you have these tools built into your IDE, please disable them for the submission.
- You are only allowed to use the libraries provided in the `requirements.txt` file and packages provided in the python standard library (e.g. `os` and `typing` etc).
- Your code must run 'as submitted', with no additional packages or software required.
- You can use the internet to access package documentation, StackOverflow answers etc, but try to write the code yourself and avoid copying and pasting large chunks of code.

## Guidance

- Follow Git best practices throughout your development.
- Try not to spend too much time on the task.
- Interact with your implementation for testing using a notebook like interface. The provided `dev.py` can be used with [VS Code](https://code.visualstudio.com/docs/python/jupyter-support-py) to allow a Jupyter-like workflow, without needing to use a notebook file. *DO NOT* commit any Jupyter notebook files with this repository.
- Passing all the unit tests does not guarantee a correct and robust implementation.
- If your machine is not powerful enough to run the GPT2 medium model, you can try to develop using [the small model](https://huggingface.co/openai-community/gpt2) by changing `"openai-community/gpt2-medium"` to `"openai-community/gpt2"` in the `__init__` function. The unit tests are unlikely to pass with this model. Make sure to change the model back upon submission. Another option is to use a service like Google Colab which should be able to use the models.