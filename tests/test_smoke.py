import webshart


def test_hello():
    """Test the hello function works."""
    assert webshart.hello() == "Hello from Webshart!"


def test_version():
    """Test version is accessible."""
    assert webshart.__version__ == "0.1.0"

print(f"welcome to webshart version {webshart.__version__}")