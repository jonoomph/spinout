# shaders.py
vertex_shader = '''
#version 330
in vec3  in_vert;
in vec4  in_color;
in vec3  in_tangent;    // new attribute: per-vertex tangent
uniform mat4 mvp;
uniform vec3 cam_pos;
out vec4  color;
out float dist;
out vec3  v_pos;
out vec3  v_tangent;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    color      = in_color;
    dist       = length(in_vert - cam_pos);
    v_pos      = in_vert;
    v_tangent  = normalize(in_tangent);
}
'''

fragment_shader = '''
#version 330
in vec4  color;
in float dist;
in vec3  v_pos;

uniform float wetness;
uniform float fog_density;
uniform vec3  fog_color;
uniform float noise_scale;  // >0 asphalt/gravel, <0 concrete

out vec4 fragColor;

// simple 2D value-noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453);
}
float valueNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0 - 2.0*f);
    float a = hash(i + vec2(0,0));
    float b = hash(i + vec2(1,0));
    float c = hash(i + vec2(0,1));
    float d = hash(i + vec2(1,1));
    return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}
// 4-octave fractal Brownian motion
float fbm(vec2 p) {
    float v = 0.0, amp = 0.6;
    for(int i = 0; i < 4; i++) {
        v   += amp * valueNoise(p);
        p   *= 2.0;
        amp *= 0.5;
    }
    return v;
}

void main() {
    float scale = abs(noise_scale);
    // pick a slightly lower wet-shine on concrete
    float wetAmt = wetness * 0.3;
    vec3 base = color.rgb * (1.0 + wetAmt);

    // world-space XZ noise on every surface
    vec2 P = v_pos.xz * scale * 0.1;
    float n = fbm(P) - 0.5;
    // tweak amplitude so concrete gets a bit bolder
    float amp = (noise_scale < 0.0) ? 0.35 : 0.25;
    base *= 1.0 + n * amp;

    // fog blend
    float f = exp(-dist * fog_density);
    fragColor = vec4(mix(fog_color, base, f), color.a);
}
'''

vertex_shader_lit = '''
#version 330
in vec3 in_vert;
in vec3 in_normal;
in vec4 in_color;
uniform mat4 mvp;
uniform vec3 light_dir;
uniform vec3 cam_pos;
uniform float wetness;
out vec4 color;
out float dist;
out vec3 v_pos;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    vec3 norm = normalize(in_normal);
    vec3 light = normalize(light_dir);
    float diff = max(dot(norm, light), 0.0);
    float ambient = 0.2;
    vec3 base_col = in_color.rgb * (1.0 + wetness * 0.3);
    vec3 base = base_col * (ambient + diff * (1.0 - ambient));
    vec3 view = normalize(cam_pos - in_vert);
    vec3 reflect_dir = reflect(-light, norm);
    float spec = pow(max(dot(view, reflect_dir), 0.0), 32.0) * wetness * 8.0;
    color = vec4(base + spec, in_color.a);
    dist = length(in_vert - cam_pos);
    v_pos = in_vert;
}
'''

fragment_shader_lit = '''
#version 330
in vec4  color;
in float dist;
in vec3  v_pos;

uniform float fog_density;
uniform vec3  fog_color;
uniform float noise_scale;

out vec4 fragColor;

// value-noise helper
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453);
}
float valueNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0 - 2.0*f);
    float a = hash(i + vec2(0,0));
    float b = hash(i + vec2(1,0));
    float c = hash(i + vec2(0,1));
    float d = hash(i + vec2(1,1));
    return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}
// 4-octave fractal Brownian motion
float fbm(vec2 p) {
    float v = 0.0, amp = 0.6;
    for(int i = 0; i < 4; i++) {
        v   += amp * valueNoise(p);
        p   *= 2.0;
        amp *= 0.5;
    }
    return v;
}

void main() {
    // take the lit color from the vertex stage
    vec3 litCol = color.rgb;

    // apply the same fBm noise as unlit shader
    float scale = abs(noise_scale);
    vec2 P     = v_pos.xz * scale * 0.1;
    float n    = fbm(P) - 0.5;
    // give concrete a slightly stronger bumpiness
    float amp  = (noise_scale < 0.0) ? 0.35 : 0.25;
    litCol     *= 1.0 + n * amp;

    // fog
    float f = exp(-dist * fog_density);
    fragColor = vec4(mix(fog_color, litCol, f), color.a);
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

vertex_shader_tex = '''
#version 330
in vec3 in_vert;
in vec2 in_tex;
uniform mat4 mvp;
uniform vec3 cam_pos;
out vec2 v_tex;
out float dist;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_tex = in_tex;
    dist = length(in_vert - cam_pos);
}
'''

fragment_shader_tex = '''
#version 330
in vec2 v_tex;
in float dist;
uniform sampler2D tex;
uniform float wetness;
uniform float fog_density;
uniform vec3 fog_color;
out vec4 fragColor;
void main() {
    vec4 color = texture(tex, v_tex);
    if (color.a < 0.1) discard;
    vec3 base = color.rgb * (1.0 + wetness * 0.3);
    float fog = exp(-dist * fog_density);
    fragColor = vec4(mix(fog_color, base, fog), color.a);
}
'''

def create_shaders(ctx):
    try:
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        prog_lit = ctx.program(vertex_shader=vertex_shader_lit, fragment_shader=fragment_shader_lit)
        prog2d = ctx.program(vertex_shader=vertex_shader_2d, fragment_shader=fragment_shader_2d)
        prog_tex = ctx.program(vertex_shader=vertex_shader_tex, fragment_shader=fragment_shader_tex)
        print("Shaders compiled successfully")
        return prog, prog2d, prog_lit, prog_tex
    except Exception as e:
        print(f"Shader compilation error: {e}")
        raise
