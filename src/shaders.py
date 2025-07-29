# shaders.py
vertex_shader = '''
#version 330
in vec3 in_vert;
in vec4 in_color;
uniform mat4 mvp;
out vec4 color;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    color = in_color;
}
'''

fragment_shader = '''
#version 330
in vec4 color;
out vec4 fragColor;
void main() {
    fragColor = color;
}
'''

vertex_shader_2d = '''
#version 330
in vec2 in_pos;
in vec2 in_tex;
out vec2 v_tex;
void main() {
    gl_Position = vec4(in_pos * 2.0 - 1.0, 0.0, 1.0); // NDC
    v_tex = in_tex;
}
'''

fragment_shader_2d = '''
#version 330
in vec2 v_tex;
uniform sampler2D tex;
out vec4 fragColor;
void main() {
    vec4 color = texture(tex, v_tex);
    if (color.a < 0.1) discard;
    fragColor = color;
}
'''

def create_shaders(ctx):
    try:
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        prog2d = ctx.program(vertex_shader=vertex_shader_2d, fragment_shader=fragment_shader_2d)
        print("Shaders compiled successfully")
        return prog, prog2d
    except Exception as e:
        print(f"Shader compilation error: {e}")
        raise