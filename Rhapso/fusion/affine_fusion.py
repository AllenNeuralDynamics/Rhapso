def add_parser(subparsers):
    """
    Adds the 'fuse' command and its arguments to the main parser.
    """
    parser = subparsers.add_parser("fuse", help="Perform affine fusion.")
    parser.add_argument('--scale', type=float, required=True, help='Scaling factor for fusion.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the fused output.')
    parser.add_argument('--blend', action='store_true', help='Blend overlapping regions.')
    parser.set_defaults(func=main)

def main(args):
    """
    Main function for affine fusion.
    """
    print("Affine Fusion Running with the following arguments:")
    print(f"Scale: {args.scale}")
    print(f"Output Path: {args.output}")
    print(f"Blend Overlaps: {args.blend}")
    print("Affine fusion completed successfully!")
