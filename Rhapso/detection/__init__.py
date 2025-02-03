def main(args):
    print("Hello, Detection!")
    print("Received the following arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
