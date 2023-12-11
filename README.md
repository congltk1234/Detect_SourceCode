# Detect_SourceCode

`git clone https://github.com/congltk1234/Detect_SourceCode.git`


`venv/Scripts/Activate.ps1`


`pip install -r requirements.txt --use-deprecated=legacy-resolver`

`uvicorn main:app --reload`


# Run dockerize
`docker build -t myimage .`

`docker run -d --name mycontainer -p 80:80 myimage`

# Test
`pytest app/tests/test_detect.py -W ignore::DeprecationWarning`
`pytest -p no:warnings`


# To exe
`Pyinstaller --noconfirm main.py --collect-data guesslang --copy-metadata guesslang --copy-metadata tqdm --copy-metadata regex --copy-metadata requests --copy-metadata packaging --copy-metadata filelock --copy-metadata numpy --copy-metadata huggingface-hub --copy-metadata safetensors --copy-metadata pyyaml --onefile`
https://stackoverflow.com/questions/60157335/cant-pip-install-tensorflow-msvcp140-1-dll-missing 