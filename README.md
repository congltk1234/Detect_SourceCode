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

https://stackoverflow.com/questions/60157335/cant-pip-install-tensorflow-msvcp140-1-dll-missing 