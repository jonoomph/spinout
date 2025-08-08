# shaders.py
vertex_shader = '''
#version 330
in vec3  in_vert;
in vec4  in_color;
uniform mat4 mvp;
uniform vec3 cam_pos;
uniform float point_size;
out vec4  color;
out float dist;
out vec3  v_pos;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    gl_PointSize = point_size;
    color      = in_color;
    dist       = length(in_vert - cam_pos);
    v_pos      = in_vert;
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
uniform int terrain_mode;   // 0 none, 1 grass, 2 dirt/sand, 3 snow

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
    float wetAmt = wetness * 0.2;
    vec3 base = color.rgb * (1.0 + wetAmt);

    // terrain textures
    vec2 texP = v_pos.xz * 0.1;
    if (terrain_mode == 1) {
        float g = fbm(texP * 1.5);
        base *= 0.8 + 0.4 * g;
    } else if (terrain_mode == 2) {
        float g = valueNoise(texP * 8.0);
        base *= 0.9 + 0.1 * g;
    } else if (terrain_mode == 3) {
        float g = fbm(texP * 2.0);
        base = base * (0.9 + 0.1 * g) + vec3(g * 0.15);
    }

    if (scale > 0.0) {
        // world-space XZ noise for roads
        vec2 P = v_pos.xz * scale * 0.1;
        float n = fbm(P) - 0.5;
        // tweak amplitude so concrete gets a bit bolder
        float amp = (noise_scale < 0.0) ? 0.35 : 0.25;
        base *= 1.0 + n * amp;
        if (noise_scale < 0.0) {
            base *= 0.9;
        }
    }

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
uniform vec3 light_color;
uniform vec3 cam_pos;
uniform float wetness;

out vec4 color;
out float dist;
out vec3 v_pos;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);

    vec3 N = normalize(in_normal);
    vec3 L = normalize(light_dir);
    float NdotL = max(dot(N, L), 0.0);

    // Wetness: slightly darker diffuse when wet (realistic) instead of brighter
    float wet = clamp(wetness, 0.0, 1.0);
    vec3 albedo = in_color.rgb * mix(1.0, 0.85, wet);

    // Basic lambert + ambient
    float ambient = 0.25;
    vec3 lambert = albedo * (ambient + 0.75 * NdotL) * light_color;

    // View-dependent spec highlight scaled by wetness (kept gentle)
    vec3 V = normalize(cam_pos - in_vert);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 16.0) * (0.15 + 0.25 * wet);

    // Cheap fresnel to avoid plastic look; small contribution
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0) * (0.20 * wet);

    vec3 col = lambert + (spec + fresnel) * light_color;

    color = vec4(min(col, vec3(1.0)), in_color.a);  // linear color out
    dist  = length(in_vert - cam_pos);
    v_pos = in_vert;
}
'''

fragment_shader_lit = '''
#version 330
in vec4  color;   // linear lit color from VS
in float dist;
in vec3  v_pos;

uniform float wetness;
uniform float fog_density;
uniform vec3  fog_color;
uniform float noise_scale;   // >0 asphalt/gravel, <0 concrete
uniform int   terrain_mode;  // 0 none, 1 grass, 2 dirt/sand, 3 snow

// Optional controls (safe defaults if you don't set them)
uniform float grade_min_y;   // default: -10 if unset
uniform float grade_max_y;   // default:  50 if unset
uniform float micro_scale;   // default:  16 if unset

out vec4 fragColor;

// ---- noise helpers ----
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float valueNoise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    f=f*f*(3.0-2.0*f);
    float a=hash(i+vec2(0,0));
    float b=hash(i+vec2(1,0));
    float c=hash(i+vec2(0,1));
    float d=hash(i+vec2(1,1));
    return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}
float fbm(vec2 p){
    float v=0.0, amp=0.6;
    for(int i=0;i<4;i++){ v+=amp*valueNoise(p); p*=2.0; amp*=0.5; }
    return v;
}

// exp2 fog + mild height bias so ground hazes a touch more
float fogFactor(float d, float density, float y){
    float dens = max(density, 0.0);
    float heightBias = clamp(0.20 * exp(-0.02 * max(y, 0.0)), 0.0, 0.20);
    float dd = d * (1.0 + heightBias);
    float f = exp(-dens * dens * dd * dd);
    return clamp(f, 0.0, 1.0);
}

void main() {
    vec3 col = color.rgb;

    // --- terrain texturing (subtle) ---
    vec2 texP = v_pos.xz * 0.1;
    if (terrain_mode == 1) {           // grass
        float g = fbm(texP * 1.5);
        col *= 0.85 + 0.35 * g;
    } else if (terrain_mode == 2) {    // dirt/sand
        float g = valueNoise(texP * 8.0);
        col *= 0.92 + 0.10 * g;
    } else if (terrain_mode == 3) {    // snow
        float g = fbm(texP * 2.0);
        // less additive brightening to avoid washout
        col = col * (0.92 + 0.08 * g) + vec3(g * 0.05);
        // if wet snow: reduce brightness a bit and compress highlights
        float w = clamp(wetness, 0.0, 1.0);
        if (w > 0.0) {
            col = mix(col, col * 0.9, w * 0.5);       // darken slightly when wet
            col = col / (1.0 + 0.6 * w * col);        // mild tone-map to tame blowouts
        }
    }

    // --- road macro noise ---
    float scale = abs(noise_scale);
    if (scale > 0.0) {
        vec2 P  = v_pos.xz * scale * 0.1;
        float n = fbm(P) - 0.5;
        float amp = (noise_scale < 0.0) ? 0.30 : 0.22; // concrete a bit stronger
        col *= 1.0 + n * amp;
        if (noise_scale < 0.0) col *= 0.92;
    }

    // --- height-based grade (subtle) ---
    float gmin = (grade_min_y == 0.0 && grade_max_y == 0.0) ? -10.0 : grade_min_y;
    float gmax = (grade_min_y == 0.0 && grade_max_y == 0.0) ?  50.0 : grade_max_y;
    float t = clamp((v_pos.y - gmin) / max(gmax - gmin, 0.001), 0.0, 1.0);
    col *= mix(0.96, 1.06, t);

    // --- fog controls (fix snow behavior) ---
    float dens = fog_density;
    if (terrain_mode == 3) {
        if (wetness < 0.001) {
            dens = 0.0;          // Snow + DRY => no fog
        } else {
            dens *= 0.5;         // Snow + WET => reduce fog so it isn't a whiteout
        }
    }

    // visibility helpers (same as before, but using dens)
    float f = fogFactor(dist, dens, v_pos.y);

    // ease wet darkening with distance so surfaces don't vanish
    float wet = clamp(wetness, 0.0, 1.0);
    col *= 1.0 + wet * 0.04 * (1.0 - f);

    // tiny contrast bump for road-like surfaces under fog
    if (scale > 0.0) {
        col *= 1.0 + 0.06 * (1.0 - f);
    }

    // contrast-aware fog so dark asphalt keeps identity
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float retain_boost = (1.0 - luma) * 0.35;
    float f_adj = clamp(f + (1.0 - f) * retain_boost, 0.0, 1.0);

    // fog blend in linear, then gamma out
    vec3 outCol = mix(fog_color, col, f_adj);
    outCol = pow(max(outCol, 0.0), vec3(1.0/2.2));
    fragColor = vec4(outCol, color.a);
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
in vec3 v_pos;

uniform sampler2D tex;
uniform float wetness;
uniform float fog_density;
uniform vec3  fog_color;

// Optional controls (safe defaults if you don't set them)
uniform float grade_min_y;  // default: -10 if unset
uniform float grade_max_y;  // default:  50 if unset

out vec4 fragColor;

void main() {
    vec4 s = texture(tex, v_tex);
    if (s.a < 0.1) discard;

    float wet = clamp(wetness, 0.0, 1.0);

    // Darken diffuse slightly when wet (no normals here, so keep it simple)
    vec3 base = s.rgb * mix(1.0, 0.88, wet);

    // Height grade (subtle)
    float gmin = (grade_min_y == 0.0 && grade_max_y == 0.0) ? -10.0 : grade_min_y;
    float gmax = (grade_min_y == 0.0 && grade_max_y == 0.0) ?  50.0 : grade_max_y;
    float t = clamp((v_pos.y - gmin) / max(gmax - gmin, 0.001), 0.0, 1.0);
    base *= mix(0.97, 1.05, t);

    // exp2 fog in linear, then gamma out
    float f = exp(-fog_density * fog_density * dist * dist);
    vec3 col = mix(fog_color, base, f);
    col = pow(max(col, 0.0), vec3(1.0/2.2));
    fragColor = vec4(col, s.a);
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
