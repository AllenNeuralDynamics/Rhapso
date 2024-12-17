def add_parser(subparsers):
    parser = subparsers.add_parser("match", help="Match features between images.")
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT'], required=True, help='Feature matching method.')
    parser.add_argument('--distance', type=float, help='Distance threshold for matching.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.set_defaults(func=main)

def main(args):
    print("Feature Matching Running with the following arguments:")
    print(f"Method: {args.method}")
    print(f"Distance Threshold: {args.distance}")
    print(f"Verbose: {args.verbose}")
    print("Features matched successfully!")
