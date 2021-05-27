# NoYetSelfAware

The objective is to create my own complete implementation of a Neural Networks, with the help of numpy

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
python3 -m twine upload --repository testpypi dist/*
```

You will be prompted for a username and password.

 - For the username, use `__token__`.
 - For the password, use the token value (including the pypi- prefix).

### Installing

```sh
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps https://test.pypi.org/project/NotYetSelfAware/0.0.1/
```