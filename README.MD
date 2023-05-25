# Fast Chat API

This is a simple chat bot implementation. Data is from the public Blender chatbot dataset. The model is a GPT-2 model fine-tuned on the Blender dataset. The model is served using the FastAPI framework.

## Prequisites

- Mac command line tools

```console
xcode-select --install
```

- Install Tree:

  ```brew
    brew install tree
  ```

- install 3.8 version of python

  ```console
  brew install python@3.8
  pip3 install virtualenv
  ```

  test it with `python3` (You can leave terminal with `exit()`)

- Setup virtual environment

  - create virtual environment and use

```console
  virtualenv -p python3.8 virtual-env
```

- run

  ```console
    source virtual-env/bin/activate
    python3 --version
    which python3
    deactivate
  ```

- install dependencies

  - `pip3 install -r requirements.txt`

## Format code

```console
python3 -m black . 
```

## Run

```console
uvicorn main:app --reload
```

## Use

Access the API from your frontend: You can now make POST requests to `http://localhost:8000/process` from your frontend to send user messages and receive the generated responses. You can use libraries like axios or fetch in JavaScript to make the API requests.