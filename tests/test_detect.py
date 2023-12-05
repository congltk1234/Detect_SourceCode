import pytest

import sys
import os
# pytest app import fix
dynamic_path = os.path.abspath('.')
sys.path.append(dynamic_path)

from app.func.detect import *
from tests.test_core_fixtures import *
################################# Test #####################################################

def test_preprocess(input_text):
    """
    Test to check if the function 'preprocess' is Spliting raw text into list of blocks.
    """
    block_text,large_text = input_text
    # Check if the returned list for `block_text` has only one element
    assert isinstance(preprocess(block_text), list), "Should be a list of blocks"
    assert len(preprocess(block_text)) == 1
    # Check if the returned list for `large_text` has more than one element
    assert isinstance(preprocess(large_text), list), "Should be a list of blocks"
    assert len(preprocess(large_text)) > 1

    # Check if user input blank text
    assert preprocess('') == [''],"Expected: [''], Actual: {0}".format(preprocess(''))
    # Check if user input single character
    assert preprocess('a') == ['a'],"Expected: ['a'], Actual: {0}".format(preprocess('a'))
    # Check if user input number
    assert preprocess(0) == ['0'],"Expected: ['0'], Actual: {0}".format(preprocess(0))
    

##########################################
############ GUESSLANG ###################
##########################################
class TestGuesslang(object):
    def test_initialize_Guesslang_models(self):
        """
        Test to check if Guesslang model are loading correctly.
        """
        guesslang = initialize_Guesslang_models()
        assert guesslang is not None

    def test_guessLang(self,block_text):
        """
        Test to check if the function 'guessLang' is returning a string with the correct name.
        It also checks if the returned object is an instance of string
        """
        block = block_text
        name = guessLang(block)
        assert isinstance(name, str), "Should be a string name"
        assert name == 'python',"Expected: python, Actual: {0}".format(name)
        # Check if user input number
        assert isinstance(guessLang(0), str)
        # Check if user input blank text
        assert guessLang('') == None

    def test_guessLang_extract(self,large_text):
        """
        Test to check if the function 'test_guessLang_extract' is returning a dictionary.
        It also checks if the input are: large_Text, blank text, number
        """
        raw_text = large_text
        response = guessLang_extract(raw_text)
        assert isinstance(response, dict), "Should be a dictionary"
        # Check if user input blank text
        assert isinstance(guessLang_extract(''), dict), "Should be a dictionary"
        assert guessLang_extract('') == {'msg': 'No SourceCode found'}, 'Should return message'
        # Check if user input number
        assert isinstance(guessLang_extract(0), dict), "Should be a dictionary"
        assert guessLang_extract(0) == {'msg': 'No SourceCode found'}, 'Should return message'


##########################################
############ CODEBERT ####################
##########################################

class TestCodebert(object):
    def test_initialize_CODEBERT_models(self):
        """
        Test to check if CODEBERT model are loading correctly.
        """
        codebert = initialize_CODEBERT_models()
        assert codebert is not None


    def test_codeBERT(self, block_text):
        """
        Test to check if the function 'codeBERT' is returning a string with the correct name.
        It also checks if the returned object is an instance of string
        """
        block = block_text
        name = codeBERT(block)
        assert isinstance(name, dict), "Should be a string name"
        assert name['label'] == 'python',"Expected: python, Actual: {0}".format(name)

        # Check if user input number
        assert isinstance(codeBERT(''), dict)

        # Test that a `codeBERT` is raised when the user input others type (except string)
        with pytest.raises(ValueError) as excinfo:  
            codeBERT(0)  
        assert str(excinfo.value) == "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples)."  