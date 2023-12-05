import pytest

# ################################ Test Initialize #####################################################
# Fixture for data initialization
@pytest.fixture
def large_text():
    sample_text = open("tests/sample.txt", "r")
    sample_text = sample_text.read()
    return sample_text

@pytest.fixture
def block_text():
    sample_text =   """ n = int(input("Enter the number of terms in the Fibonacci sequence: "))
                        print("Fibonacci sequence:")
                        for i in range(n):
                            print(fibonacci(i), end=" ")
                    """
    return sample_text

@pytest.fixture
def input_text(block_text,large_text):
    return block_text,large_text