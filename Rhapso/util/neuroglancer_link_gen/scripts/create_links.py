# from ng_link import exaspim_link
# from ng_link import dispim_link
from ng_link import raw_link


def main():
    # Fill in your own data
    base_channel_path = "/Users/jonathan.wong/Projects/aind-ng-link/tester.xml"
    # cross_channel_path = "/Users/jonathan.wong/Projects/aind-ng-link/registered_tester_coreg.xml"
    s3_path = (
        "s3://aind-open-data/diSPIM_647459_2022-12-21_00-39-00/diSPIM.zarr"
    )

    # exaspim_link.generate_exaspim_link(
    #     base_channel_path,
    #     s3_path,
    #     max_dr=200,
    #     opacity=1.0,
    #     blend="default",
    #     output_json_path=".",
    # )
    # dispim_link.generate_dispim_link(
    #     base_channel_path,
    #     cross_channel_path,
    #     s3_path,
    #     max_dr=800,
    #     opacity=0.5,
    #     blend="additive",
    #     deskew_angle=-45,
    #     output_json_path=".",
    # )
    raw_link.generate_raw_link(
        base_channel_path,
        s3_path,
        max_dr=200,
        opacity=1.0,
        blend="default",
        output_json_path=".",
    )


if __name__ == "__main__":
    main()
