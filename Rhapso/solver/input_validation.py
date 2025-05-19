from Rhapso.matching.interest_point_matching import build_label_map

# This class implements an input validation process

class InputValidation:

    def __init__(self, data_global, reference_tp, registration_tp, labels, label_weights, fixed_views, group_illums,
                group_channels, group_tiles, split_timepoints, disable_fixed_views, solver_source, image_loader_data, TO_REFERENCE_TIMEPOINT, TIMEPOINTS_INDIVIDUALLY):
       
        self.data_global = data_global
        self.reference_tp = reference_tp
        self.labels = labels
        self.label_weights = label_weights
        self.fixed_views = fixed_views
        self.view_ids_global = {}
        self.fixed_view_ids = {}
        self.group_illums = group_illums
        self.group_channels = group_channels
        self.group_tiles = group_tiles
        self.split_timepoints = split_timepoints
        self.registration_tp = registration_tp
        self.disable_fixed_views = disable_fixed_views
        self.source_points = None
        self.solver_source = solver_source
        self.TO_REFERENCE_TIMEPOINT = TO_REFERENCE_TIMEPOINT
        self.TIMEPOINTS_INDIVIDUALLY = TIMEPOINTS_INDIVIDUALLY
        self.image_loader_data = image_loader_data  # Data_global and image loader are likely the same and may be able to be eliminated.

    def input_validation(self):

        self.init_registration_parameters()

        if not self.setup_parameters(self.image_loader_data, self.view_ids_global):
            return None
        
        label_map_global = build_label_map(
                self.image_loader_data, self.view_ids_global, map
            )

        # If solver_source == IP set up variables for Interestpoints as source
        if self.source_points == self.solver_source["IP"]:
            if self.labels == None or len(self.labels) == 0:
                print("No labels specified. Stopping.")
                return None
            # Map step may be able to be consolidated here
            if self.label_weights == None or len(self.label_weights) == 0:
                self.label_weights = [1.0] * len(self.labels)
            if len(self.label_weights) != len(self.labels):
                print(
                    "You need to specify as many weights as labels, or do not specify weights at all"
                )
                return None

            # String and float
            map = {}

            for i in range(len(self.labels)):
                map[self.labels[i]] = self.label_weights[i]
            print(f"labels & weights: {map}")

            # This section of conditionals may be able to be removed.
            if not self.group_illums:
                self.group_illums = False

            if not self.group_channels:
                self.group_channels = False

            if not self.group_tiles:
                self.group_tiles = False

            if not self.split_timepoints:
                self.split_timepoints = False
        else:
            print("Using stitching results as source for solve.")

            labelMapGlobal = None

            if not self.group_illums:
                self.group_illums = True

            if not self.group_channels:
                self.group_channels = True

            if not self.group_tiles:
                self.group_tiles = False

            if not self.split_timepoints:
                self.split_timepoints = False

        print("The following grouping/splitting modes are set:")
        print(f"groupIllums: {self.group_illums}")
        print(f"groupChannels: {self.group_channels}")
        print(f"groupTiles: {self.group_tiles}")
        print(f"splitTimepoints: {self.split_timepoints}")

        if not self.fixed_view_ids or len(self.fixed_view_ids) == 0:
            # fix this - Not sure if needed
            self.fixed_view_ids = self.assemble_fixed_auto()
        else:
            self.fixed_view_ids  # reset to data

        print("The following ViewIds are used as fixed views:")
        print(
            ", ".join(
                f'tpId={vid["timepoint"]} setupId={vid["setup"]}'
                for vid in self.fixed_view_ids
            )
        )

    def setup_parameters(self, data_global, view_ids_global):
        # fixed views and mapping back to original view
        if self.disable_fixed_views:
            self.fixed_view_ids = None
        else:
            # set/load fixed views
            if not self.fixed_views:
                print(
                    "First ViewId(s) will be used as fixed for each respective registration subset (e.g. timepoint) ..."
                )
                self.fixed_view_ids = None
            else:
                print("Parsing fixed ViewIds ...")
                # Todo fix
                parsed_views = self.fixed_views  # all views

                # Assuming view ids are in a list of dictionaries
                self.fixed_view_ids = data_global[view_ids_global]

                # add the rest of the dictionaries
                for view in parsed_views:
                    self.fixed_view_ids.append(view)

                if len(parsed_views) != len(self.fixed_view_ids):
                    print(
                        f"Warning: only {len(self.fixed_view_ids)} of {len(parsed_views)} that you specified as fixed views exist and are present."
                    )

                if not self.fixed_view_ids:
                    raise ValueError(
                        "Fixed views couldn't be parsed. Please provide a valid fixed view."
                    )

                print("The following ViewIds are fixed: ")
                for vid in self.fixed_view_ids:
                    print(f'tpId={vid["timepoint"]} setupId={vid["setup"]}')

        return True

    # Check if reference timepoint
    # sd doesnt get used
    # We might be able to trim this down to always just have the first view be the fixed point.
    def assemble_fixed_auto(self, reference_tp):
        fixed = set()

        self.view_ids_global.sort()

        #  double check registrations types --> list of Ints
        if self.registration_tp == self.TO_REFERENCE_TIMEPOINT:
            for view_id in self.view_ids_global:
                if view_id["timepoint"] == self.reference_tp:
                    fixed.add(view_id)
                    break
        elif self.registration_tp == self.TIMEPOINTS_INDIVIDUALLY:
            # it is sorted by timepoint
            fixed.add(self.view_ids_global[0])
            current_tp = self.view_ids_global[0]["timepoint"]

            for view_id in self.view_ids_global:
                # next timepoint
                if view_id["timepoint"] != current_tp:
                    fixed.add(view_id)
                    current_tp = view_id["timepoint"]
        else:
            fixed.add(self.view_ids_global[0])  # always the first view is fixed

        return fixed

    def init_registration_parameters(self):
        # Retrieves data_global object

        if not self.image_loader_data:
            raise ValueError("Couldn't load SpimData XML project.")

        # Should have timepoint and set up
        view_ids_global = self.image_loader_data["view_ids"]
        # retrieves just viewIds

        if not view_ids_global or len(view_ids_global) == 0:
            raise ValueError("No ViewIds found.")

        self.reference_tp = None  # Assuming referenceTP is initially None
        registration_tp = (
            "TO_REFERENCE_TIMEPOINT"  # Assuming registrationTP is set to this value
        )

        if not self.reference_tp:
            # Get the timepoint of the first viewId in the list
            self.reference_tp = view_ids_global[0]["timepoint"]
        else:
            # Retrieve and sort all timepoints
            timepoint_to_process_list = []

            for view in self.image_loader_data:
                # This could be viewId[0]
                timepoint_to_process_list.append(view["timepoint"])
            for view in view_ids_global:
                timepoint_to_process_list.append(view["timepoint"])

            timepoint_to_process_set = set(sorted(timepoint_to_process_list))

            if self.reference_tp not in timepoint_to_process_set:
                raise ValueError(
                    "Specified reference timepoint is not part of the ViewIds that are processed."
                )
        # I don't think this gets hit since it gets set to null in this case. May need to follow up about params.

    def run(self):
        self.input_validation()
