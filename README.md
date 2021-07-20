# NoYetSelfAware

A beautiful library for DeeplLearning, with the help of numpy

![](./assets/demo.gif)

## About

This project is the occasion to reimplement my theoric learning as it grows.

# Summary

### Legend
|||
|-|-|
|⬜|To Do|
|🚧|Work in Progress|
|☑️|Done|

|Activation||
|:-|-:|
|leakyReLU|☑️|
|ReLU|☑️|
|sigmoid|☑️|
|tanh|☑️|

|Layers||
|:-|-:|
|Dense|☑️|
|Output|☑️|
|Dropout|⬜|
|Convolution|⬜|
|Long Short Term Memory|⬜|

|Optimizers||
|:-|-:|
|Gradient Descent|☑️|
|MiniBatch Gradient Descent|☑️|
|Stochastic Gradient Descent|☑️|
|Momentum|☑️|
|RMSprop|☑️|
|Adam|☑️|

|Validation||
|:-|-:|
|Accuracy|☑️|
|Recall|⬜|
|F1_score|⬜|

|Cost||
|:-|-:|
|Binary Cross Entropy|☑️|
|Mean Square Error|⬜|
|Soft Max|⬜|

|Regularization||
|:-|-:|
|L1|⬜|
|L2|⬜|

|Learning Rate Decay||
|:-|-:|
|exponential decay|⬜|
|staircase decay|⬜|

|Others||
|:-|-:|
|Batch Normalization|⬜|

# PyPi Package

## Installation

```sh
pip install NoYetSelfAware
```

or 

```sh
python3 -m pip install NoYetSelfAware
```

## Package

It's currently my first python package.

It was done following this nice tutorial: [https://packaging.python.org/tutorials/packaging-projects/](https://packaging.python.org/tutorials/packaging-projects/)

### Build


Make sure you have the latest versions of PyPA’s build installed:

```sh
python3 -m pip install --upgrade build
```

Now run this command in the root of the project:

```sh
python3 -m build
```

This command should output a lot of text and once completed should generate two files in the dist directory:
```
dist/
  NoYetSelfAware-$VERSION-py3-none-any.whl
  NoYetSelfAware-$VERSION.tar.gz
```

### Upload to pip

Now that you are registered, you can use twine to upload the distribution packages.

You’ll need to install Twine:

```sh
python3 -m pip install --user --upgrade twine
```

Once installed, run Twine to upload all of the archives under `dist`:

```sh
twine upload dist/*
```

You will be prompted for a username and password.

 - For the username, use `__token__`.
 - For the password, use the token value (including the pypi- prefix).
