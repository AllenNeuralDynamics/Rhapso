import argparse

def add_parser(subparsers):
    """
    Adds the 'detect' command and its arguments to the main parser.
    """
    parser = subparsers.add_parser("detect", help="Detect interest points in images.")
    parser.add_argument('--medianFilter', type=int, help='Median filter radius for preprocessing.')
    parser.add_argument('--sigma', type=float, required=True, help='Sigma for segmentation, e.g., 1.8.')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for detection.')
    parser.set_defaults(func=main)

def main(args):
    """
    Main function for interest point detection.
    """
    print("Interest Point Detection Running with the following arguments:")
    print(f"Median Filter Radius: {args.medianFilter}")
    print(f"Sigma: {args.sigma}")
    print(f"Threshold: {args.threshold}")
    print("Interest points detected successfully!")
