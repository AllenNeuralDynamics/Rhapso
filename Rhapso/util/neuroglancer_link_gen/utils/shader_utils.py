"""
Module containing utility functions to generate neuroglancer shader code
"""
from typing import Tuple


def create_monochrome_shader(color: str, emitter: str, vec: str) -> str:
    """
    Creates a configuration for the neuroglancer shader.
    This shader generates a monochromatic image.

    Parameters
    ------------------------
    color:  str
        The color of the monochromatic image
    emitter: str
        suffix of WebGL emit* method being called (e.g. 'RGB')
    vec: str
        class of vector we want color returned as in WebGL
        (e.g. 'vec3')

    Returns
    ------------------------
    str
        String with the shader configuration for neuroglancer.
    """

    # Add all necessary ui controls here
    ui_controls = [
        f'#uicontrol {vec} color color(default="{color}")',
        "#uicontrol invlerp normalized",
    ]

    # color emitter
    emit_color = (
        "void main() {\n" + f"emit{emitter}(color * normalized());" + "\n}"
    )
    shader_string = ""

    for ui_control in ui_controls:
        shader_string += ui_control + "\n"

    shader_string += emit_color

    return shader_string


def create_rgb_shader(
    r_range: Tuple[int, int],
    g_range: Tuple[int, int],
    b_range: Tuple[int, int],
) -> str:
    """
    Return shader code for an RGB image with different dynamic
    ranges for each channel.

    Parameters
    ----------
    r_range: Tuple[int, int]
        Dynamic range of the R channel (min, max)
    g_range: Tuple[int, int]
        Dynamic range of the G channel (min, max)
    b_range: Tuple[int, int[
        Dynamic range of the B channel (min, max)

    Returns
    -------
    str
        String containing the shader code for a neuroglancer
        RGB image.
    """

    code = "#uicontrol invlerp "
    code += f"normalized_r(range=[{r_range[0]}, {r_range[1]}])\n"
    code += "#uicontrol invlerp "
    code += f"normalized_g(range=[{g_range[0]}, {g_range[1]}])\n"
    code += "#uicontrol invlerp "
    code += f"normalized_b(range=[{b_range[0]}, {b_range[1]}])\n"
    code += "void main(){\n"
    code += "float r = normalized_r(getDataValue(0));\n"
    code += "float g = normalized_g(getDataValue(1));\n"
    code += "float b = normalized_b(getDataValue(2));\n"
    code += "emitRGB(vec3(r, g, b));\n}\n"

    return code
