# TODO Correct for data structures from other detection.
class AdvancedRefinement:
    def __init__(
        self,
        result,
        store_intensities,
        max_spots,
        max_spots_per_overlap,
        to_process,
        max_interval_size,
    ):
        self.result = result  ## This is an array of [[[viewID],[Interval], [interestpoints],[intensities]]]
        self.store_intensities = store_intensities
        self.max_spots = max_spots
        self.max_spots_per_overlap = max_spots_per_overlap
        self.to_process = to_process
        self.max_interval_size = max_interval_size

        self.interest_points_per_view_id = {}
        self.intensities_per_view_id = {}
        self.intervals_per_view_id = {}


def main(self, result):

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
            inter = interval[0]
            ips = interval[1]
            intensity_list = interval[2]

            my_max_spots = round(
                self.max_spots * (sum(inter["dimensions"]) / self.max_interval_size)
            )
            if my_max_spots > 0 and my_max_spots < len(ips):
                old_size = len(ips)
                # filter points from ips, intensity_list, mymaxspots
                inter, intensities_list = filter_points(
                    inter,
                    intensity_list,
                )
                print(
                    f"filtered interval: limit "
                    + my_max_spots
                    + " old Size:"
                    + old_size
                    + "interval: "
                    + inter
                )
            else:
                print("NOT filtered interval")
            self.interest_points_per_view_id[view_id] += ips
            self.intensities_per_view_id[view_id] += intervals_list
    return (
        self.interest_points_per_view_id,
        self.intensities_per_view_id,
        self.intervals_per_view_id,
    )


def filter_points(interest_points, intensities, max_spots):
    combined_list = []
    for i in len(range(interest_points)):
        combined_list.append((intensities[i], interest_points[i]))
        print((intensities[i], interest_points[i]))

    combined_list.sort(reverse=True)
    intensities.clear()
    interest_points.clear()

    # Add back the top max_spots elements
    for i in range(min(max_spots, len(combined_list))):

        intensity, ip = combined_list[i]
        intensities.append(intensity)
        interest_points.append((ip.location))

    return interest_points, intensities


# process_results(result)
# Example usage:
# create instance of InterestPointsProcessor with appropriate arguments
# call run method on the instance


# advanced_refinements(data)
