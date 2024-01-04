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
`pyinstaller --noconfirm --onefile --console --copy-metadata "tqdm" --copy-metadata "regex" --copy-metadata "requests" --copy-metadata "packaging" --copy-metadata "filelock" --copy-metadata "numpy" --copy-metadata "huggingface-hub" --copy-metadata "safetensors" --copy-metadata "pyyaml" --collect-all "transformers" --hidden-import "transformers" --hidden-import "tokenizers" --collect-all "tokenizers" --hidden-import "tokenizers.decoders" --collect-all "tokenizers.decoders" --collect-all "guesslang"  "C:/Users/st_cong/Desktop/Detect_SourceCode/main.py"`


auto-py-to-exe

https://pypi.org/project/auto-py-to-exe/
https://stackoverflow.com/questions/60157335/cant-pip-install-tensorflow-msvcp140-1-dll-missing 