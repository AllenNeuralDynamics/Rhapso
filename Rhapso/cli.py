import argparse
import Rhapso.matching
import Rhapso.detection

def main():
    parser = argparse.ArgumentParser(
        description="Rhapso CLI - Image processing tool for detection, matching, solving, and fusion.",
        add_help=False  # Disables the default help to allow for custom help handling
    )

    # Define global commands that are not specific to subparsers
    global_commands = [
        {'flags': ['-v', '--version'], 'action': 'version', 'help': 'Show the version of Rhapso', 'version': 'Rhapso version 1.0'},
        {'flags': ['-h', '--help'], 'action': 'help', 'help': 'Show this help message and exit'},
        {'flags': ['--dryRun'], 'action': 'store_true', 'help': "Perform a 'dry run', i.e. do not save any results", 'default': False}
    ]

    # Create subparsers for the different commands
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Add subparsers for detect, match, etc...
    detect_parser = subparsers.add_parser('detect', help='Run detection commands')
    match_parser = subparsers.add_parser('match', help='Run matching commands')

    # Define common commands that appear in every subparser
    common_commands = [
        {'flags': ['-x', '--xml'], 'type': str, 'help': 'Path to the existing BigStitcher project XML', 'required': True}
    ]

    # Define detection commands
    detection_commands = [
        {'flags': ['-l', '--label'], 'type': str, 'help': 'Label of the interest points used for detection.', 'required': True},
        {'flags': ['-s', '--sigma'], 'type': float, 'help': 'Sigma value for the DOG filter.', 'required': True},
        {'flags': ['-t', '--threshold'], 'type': float, 'help': 'Threshold value for the DOG filter.', 'required': True},
    ]

    # Define matching commands
    matching_commands = [
        {'flags': ['-l', '--label'], 'type': str, 'help': 'Labels of the interest points used for registration.', 'required': True},
        {'flags': ['-m', '--method'], 'type': str, 'choices': ['FAST_ROTATION', 'FAST_TRANSLATION', 'PRECISE_TRANSLATION', 'ICP'], 'help': 'The matching method.', 'required': True},
        {'flags': ['-s', '--significance'], 'type': float, 'help': 'How much better the first match has to be compared to the second best.', 'default': 3.0},
        {'flags': ['-sr', '--searchRadius'], 'type': float, 'help': 'Search range for corresponding points in global coordinate space (only for PRECISE_TRANSLATION).', 'default': None},
        {'flags': ['-r', '--redundancy'], 'type': int, 'help': 'The redundancy of the local descriptor.', 'default': 1},
        {'flags': ['-n', '--numNeighbors'], 'type': int, 'help': 'The number of neighboring points used to build the local descriptor (only for PRECISE_TRANSLATION).', 'default': 3},
        {'flags': ['--clearCorrespondences'], 'action': 'store_true', 'help': 'Clear existing corresponding interest points before adding new ones.', 'default': False},
        {'flags': ['--matchAcrossLabels'], 'action': 'store_true', 'help': 'Match in between label classes if more than one label is specified.', 'default': False},
        {'flags': ['-ipfr', '--interestPointsForReg'], 'type': str, 'choices': ['OVERLAPPING_ONLY', 'ALL'], 'help': 'Which interest points to use for pairwise registrations.', 'default': 'ALL'},
        {'flags': ['-vr', '--viewReg'], 'type': str, 'choices': ['OVERLAPPING_ONLY', 'ALL_AGAINST_ALL'], 'help': 'Which views to register with each other.', 'default': 'OVERLAPPING_ONLY'},
        {'flags': ['--interestPointMergeDistance'], 'type': float, 'help': 'Merge interest points within this radius in px.', 'default': 5.0},
        {'flags': ['--groupIllums'], 'action': 'store_true', 'help': 'Group all illumination directions that belong to the same angle/channel/tile/timepoint together as one view.', 'default': False},
        {'flags': ['--groupChannels'], 'action': 'store_true', 'help': 'Group all channels that belong to the same angle/illumination/tile/timepoint together as one view.', 'default': False},
        {'flags': ['--groupTiles'], 'action': 'store_true', 'help': 'Group all tiles that belong to the same angle/channel/illumination/timepoint together as one view.', 'default': False},
        {'flags': ['--splitTimepoints'], 'action': 'store_true', 'help': 'Group all angles/channels/illums/tiles that belong to the same timepoint as one view.', 'default': False},
        {'flags': ['-rit', '--ransacIterations'], 'type': int, 'help': 'Max number of RANSAC iterations.', 'default': 10000},
        {'flags': ['-rme', '--ransacMaxError'], 'type': float, 'help': 'RANSAC max error in pixels.', 'default': 5.0},
        {'flags': ['-rmir', '--ransacMinInlierRatio'], 'type': float, 'help': 'RANSAC min inlier ratio.', 'default': 0.1},
        {'flags': ['-rmif', '--ransacMinInlierFactor'], 'type': float, 'help': 'RANSAC min inlier factor.', 'default': 3.0},
        {'flags': ['-ime', '--icpMaxError'], 'type': float, 'help': 'ICP max error in pixels.', 'default': 5.0},
        {'flags': ['-iit', '--icpIterations'], 'type': int, 'help': 'Max number of ICP iterations.', 'default': 200},
        {'flags': ['--icpUseRANSAC'], 'action': 'store_true', 'help': 'ICP uses RANSAC at every iteration to filter correspondences.', 'default': False}
    ]

    solver_commands = []

    fusion_commands = []

    # Function to add flags to a parser
    def add_flags_to_parser(parser, flags):
        for flag in flags:
            kwargs = {
                'action': flag.get('action', 'store'), 
                'help': flag['help'], 
                'default': flag.get('default')
            }

            if 'type' in flag:
                kwargs['type'] = flag['type']
            if 'choices' in flag:
                kwargs['choices'] = flag['choices']
            if 'required' in flag:
                kwargs['required'] = flag.get('required', False)
            if 'version' in flag:
                kwargs['version'] = flag.get('version')

            parser.add_argument(*flag['flags'], **kwargs)

    # Add flags to the parsers
    add_flags_to_parser(parser, global_commands)
    add_flags_to_parser(detect_parser, common_commands + detection_commands)
    add_flags_to_parser(match_parser, common_commands + matching_commands)

    # Parse the arguments
    args = parser.parse_args()

    # Handle the parsed arguments in a general way
    run(args)

def run(args):
    # If the command is match, call the matching file
    if args.command == 'match':
        Rhapso.matching.main(args)
    # If the command is detect, call the detection file
    elif args.command == 'detect':
        Rhapso.detection.main(args)
    else:
        print(f"Nothing is running because the following things were not specified: [match, detect, etc]")

if __name__ == "__main__":
    main()
