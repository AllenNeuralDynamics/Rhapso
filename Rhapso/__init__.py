import sys

def say_hello(name="World"):
    return f"Hello, {name} from Rhapso!"

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "World"
    print(say_hello(name))

if __name__ == "__main__":
    main()
