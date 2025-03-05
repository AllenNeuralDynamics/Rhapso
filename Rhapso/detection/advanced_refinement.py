# TODO Correct for data structures from other detection.


# See testfile for currect starting data structures.
class AdvancedRefinement:
    def __init__(self, result, to_process, max_spots):
        self.result = result  ## This is an array of [[[viewID],[Interval], [interestpoints],[intensities]]]
        self.store_intensities = False
        self.max_spots = max_spots
        self.max_spots_per_overlap = False
        self.to_process = to_process
        self.max_interval_size = 0
        self.interest_points_per_view_id = {}
        self.intensities_per_view_id = {}
        self.intervals_per_view_id = {}


def consolidate_interest_points(self, result):
    # Add ips, intervals and intensities to hashmap with viewID as key
    for view in result:
        view_id = tuple(view[0])
        points = view[2]
        interval = view[1]
        intensities = view[3]

        if points and len(points) > 0:
            if view_id not in self.interest_points_per_view_id:
                self.interest_points_per_view_id[view_id] = points

        if view_id not in self.intervals_per_view_id:
            self.intervals_per_view_id[view_id] = interval

        if self.store_intensities or self.max_spots > 0:
            if view_id not in self.intensities_per_view_id:
                self.intensities_per_view_id[view_id] = intensities

    view_ids = sorted(list(self.interest_points_per_view_id.keys()))

    # max_spots_overlap is a set param
    if self.max_spot_per_overlap and self.max_spots > 0:
        ips_list = []
        intensities_list = []
        intervals_list = []

        for id in view_ids:

            ips_list.append(self.interest_points_per_view_id[id])
            intensities_list.append(self.intensities_per_view_id[id])
            intervals_list.append(self.intervals_per_view_id[id])
            interval_data = []

            for pair in self.to_process:
                if pair[0] == list(id):
                    to_process_interval = pair[1]
                    ips_block = []
                    intensities_block = []

                    for l in range(len(ips_list) - 1):

                        block_interval = intervals_list[l]

                        if (
                            block_interval in interval_data
                            and to_process_interval in interval_data
                        ):

                            ips_block.extend(ips_list[l])
                            intensities_block.extend(intensities_list[l])
                    interval_data.append(
                        (to_process_interval, ips_block, intensities_block)
                    )

        # To later put back into interest_points_per_view_id and intensities_per_view_id
        self.interest_points_per_view_id[view_id].clear()
        self.intensities_per_view_id[view_id].clear()

        for interval in interval_data:
            intervals = interval[0]
            ips = interval[1]
            intensity_list = interval[2]

            my_max_spots = round(
                self.max_spots * (sum(intervals["dimensions"]) / self.max_interval_size)
            )
            if my_max_spots > 0 and my_max_spots < len(ips):
                old_size = len(ips)
                # filter points from ips, intensity_list, mymaxspots
                intervals, intensities_list = filter_points(
                    ips,
                    intensity_list,
                )
                print(
                    f"filtered interval: limit "
                    + my_max_spots
                    + " old Size:"
                    + old_size
                    + "interval: "
                    + intervals
                )
            else:
                print("NOT filtered interval")
            self.interest_points_per_view_id[view_id] += ips
            self.intensities_per_view_id[view_id] += intensities_list
    return (
        self.interest_points_per_view_id,
        self.intensities_per_view_id,
        self.intervals_per_view_id,
    )


def filter_points(interest_points, intensities, max_spots):
    combined_list = []
    for i in range(len(interest_points)):
        combined_list.append((intensities[i], interest_points[i]))
        print((intensities[i], interest_points[i]))

    combined_list.sort(reverse=True)
    intensities.clear()
    interest_points.clear()

    # Add back the top max_spots elements
    for i in range(max_spots):
        intensity, ip = combined_list[i]
        intensities.append(intensity)
        interest_points.append((ip))

    return interest_points, intensities


def run(self):
    self.AdvancedRefinement()
    self.consolidate_interest_point(self.data)
    return self.to_process, (
        self.interest_points_per_view_id,
        self.intensities_per_view_id,
        self.intervals_per_view_id,
    )
