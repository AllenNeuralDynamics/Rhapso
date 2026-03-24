class VolumeSampler:
    """
    Generates slices of cell_size
    respecting output_volume_size boundary.
    """

    def __init__(
        self,
        output_volume_size: tuple[int, int, int],
        cell_size: tuple[int, int, int]
    ):
        """
        Store arguments
        """
        self.output_volume_size: tuple[int, int, int] = output_volume_size
        self.cell_size: tuple[int, int, int] = cell_size

    def __iter__(self):
        """
        Cell metadata generator.
        Returns cell coordinates as well as tile id information.
        """
        oz, oy, ox = self.output_volume_size
        cz, cy, cx = self.cell_size

        for z in range(0, oz, cz):
            for y in range(0, oy, cy):
                for x in range(0, ox, cx):
                    curr_cell: geometry.AABB = \
                    (z, min(z + cz, oz),
                     y, min(y + cy, oy),
                     x, min(x + cx, ox))

                    yield curr_cell