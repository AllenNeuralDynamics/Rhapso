import argparse
from Rhapso.detection import interest_points
from Rhapso.fusion import affine_fusion
from Rhapso.matching import feature_matching
from Rhapso.solving import solver  # Import the solver module

def main():
    parser = argparse.ArgumentParser(
        description="Rhapso CLI - Image processing tool for detection, matching, solving, and fusion."
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # Dynamically add parsers from each module
    interest_points.add_parser(subparsers)
    affine_fusion.add_parser(subparsers)
    feature_matching.add_parser(subparsers)
    solver.add_parser(subparsers)  # Register the 'solve' command here
    
    # Parse arguments and call the corresponding function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
