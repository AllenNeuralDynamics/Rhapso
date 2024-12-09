import unittest
from Rhapso import say_hello

class TestRhapso(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(say_hello(), "Hello from Rhapso!")

if __name__ == "__main__":
    unittest.main()
