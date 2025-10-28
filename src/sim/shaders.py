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
    color = in_color;
    dist  = length(in_vert - cam_pos);
    v_pos = in_vert;
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
uniform float rain_strength;
uniform float rain_time;

out vec4 fragColor;

// simple value noise
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

void main() {
    vec3 base = color.rgb;

    // subtle terrain texturing
    vec2 texP = v_pos.xz * 0.1;
    if (terrain_mode == 1) {           // grass
        float g = fbm(texP * 1.5);
        base *= 0.8 + 0.4 * g;
    } else if (terrain_mode == 2) {    // dirt/sand
        float g = valueNoise(texP * 8.0);
        base *= 0.9 + 0.1 * g;
    } else if (terrain_mode == 3) {    // snow
        float g = fbm(texP * 2.0);
        base = base * (0.9 + 0.1 * g) + vec3(g * 0.15);
    }

    // road noise
    float scale = abs(noise_scale);
    if (scale > 0.0) {
        vec2 P = v_pos.xz * scale * 0.1;
        float n = fbm(P) - 0.5;
        float amp = (noise_scale < 0.0) ? 0.35 : 0.25; // concrete a bit stronger
        base *= 1.0 + n * amp;
        if (noise_scale < 0.0) base *= 0.9;
    }

    float rain = clamp(rain_strength, 0.0, 1.0);
    if (rain > 0.001) {
        vec2 rain_uv = v_pos.xz * (0.28 + rain * 0.60);
        float t = rain_time * (1.1 + rain * 1.1);
        vec2 flow_a = vec2(0.9, -1.3);
        vec2 flow_b = vec2(-0.6, 0.8);
        vec2 warp = vec2(
            fbm(rain_uv * 2.6 + flow_a * t * 0.20),
            fbm(rain_uv * 2.6 + flow_b * t * 0.18)
        ) - 0.5;
        rain_uv += warp * (1.5 + rain * 1.2);

        float streaks = sin(dot(rain_uv, vec2(0.8, 1.4)) * 22.0 + t * 6.0);
        float cross = sin(dot(rain_uv, vec2(-1.2, 0.6)) * 20.0 - t * 7.5);
        float sheets = sin((rain_uv.x - rain_uv.y) * 18.0 + t * 9.0);
        float jitter = valueNoise(rain_uv * 10.0 + vec2(t * 1.6, -t * 1.1)) - 0.5;
        float sparkle = fbm(rain_uv * 6.5 + vec2(-t * 0.5, t * 0.35)) - 0.5;

        float wet_dark = mix(1.0, 0.82, rain);
        base *= wet_dark;
        float glint = clamp((streaks + cross + sheets) * 0.16 + jitter * 0.55 + sparkle * 0.45, -1.0, 1.0);
        base += vec3(0.11, 0.14, 0.17) * glint * (0.12 + 0.30 * rain);
    }

    // clean exp fog toward horizon tint
    float f = exp(-fog_density * dist);
    vec3 outCol = mix(fog_color, base, f);
    fragColor = vec4(outCol, color.a);
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

    // slightly darker diffuse when wet
    float w = clamp(wetness, 0.0, 1.0);
    vec3 albedo = in_color.rgb * mix(1.0, 0.85, w);

    float ambient = 0.25;
    vec3 lambert = albedo * (ambient + 0.75 * NdotL) * light_color;

    vec3 V = normalize(cam_pos - in_vert);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 16.0) * (0.15 + 0.25 * w);
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0) * (0.20 * w);

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
uniform float rain_strength;
uniform float rain_time;

// Optional controls (defaults if unset)
uniform float grade_min_y;   // default: -10 if unset
uniform float grade_max_y;   // default:  50 if unset
uniform float micro_scale;   // default:  16 if unset

out vec4 fragColor;

// noise helpers
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

// exp2 fog with mild height bias
float fogFactor(float d, float density, float y){
    float dens = max(density, 0.0);
    float heightBias = clamp(0.20 * exp(-0.02 * max(y, 0.0)), 0.0, 0.20);
    float dd = d * (1.0 + heightBias);
    float f = exp(-dens * dens * dd * dd);
    return clamp(f, 0.0, 1.0);
}

void main() {
    vec3 col = color.rgb;

    // terrain texturing
    vec2 texP = v_pos.xz * 0.1;
    if (terrain_mode == 1) {           // grass
        float g = fbm(texP * 1.5);
        col *= 0.85 + 0.35 * g;
    } else if (terrain_mode == 2) {    // dirt/sand
        float g = valueNoise(texP * 8.0);
        col *= 0.92 + 0.10 * g;
    } else if (terrain_mode == 3) {    // snow
        float g = fbm(texP * 2.0);
        col = col * (0.92 + 0.08 * g) + vec3(g * 0.05);
        float w = clamp(wetness, 0.0, 1.0);
        if (w > 0.0) {
            col = mix(col, col * 0.9, w * 0.5);
            col = col / (1.0 + 0.6 * w * col);
        }
    }

    // road macro noise
    float scale = abs(noise_scale);
    if (scale > 0.0) {
        vec2 P  = v_pos.xz * scale * 0.1;
        float n = fbm(P) - 0.5;
        float amp = (noise_scale < 0.0) ? 0.30 : 0.22; // concrete stronger
        col *= 1.0 + n * amp;
        if (noise_scale < 0.0) col *= 0.92;
    }

    float rain = clamp(rain_strength, 0.0, 1.0);
    if (rain > 0.001) {
        vec2 rain_uv = v_pos.xz * (0.28 + rain * 0.60);
        float tt = rain_time * (1.1 + rain * 1.1);
        vec2 flow_a = vec2(0.9, -1.3);
        vec2 flow_b = vec2(-0.6, 0.8);
        vec2 warp = vec2(
            fbm(rain_uv * 2.6 + flow_a * tt * 0.20),
            fbm(rain_uv * 2.6 + flow_b * tt * 0.18)
        ) - 0.5;
        rain_uv += warp * (1.5 + rain * 1.2);

        float streaks = sin(dot(rain_uv, vec2(0.8, 1.4)) * 22.0 + tt * 6.0);
        float cross = sin(dot(rain_uv, vec2(-1.2, 0.6)) * 20.0 - tt * 7.5);
        float sheets = sin((rain_uv.x - rain_uv.y) * 18.0 + tt * 9.0);
        float jitter = valueNoise(rain_uv * 10.0 + vec2(tt * 1.6, -tt * 1.1)) - 0.5;
        float sparkle = fbm(rain_uv * 6.5 + vec2(-tt * 0.5, tt * 0.35)) - 0.5;

        float wet_dark = mix(1.0, 0.83, rain);
        col *= wet_dark;
        float glint = clamp((streaks + cross + sheets) * 0.16 + jitter * 0.55 + sparkle * 0.45, -1.0, 1.0);
        col += vec3(0.12, 0.15, 0.18) * glint * (0.12 + 0.30 * rain);
    }

    // height grade (subtle)
    float gmin = (grade_min_y == 0.0 && grade_max_y == 0.0) ? -10.0 : grade_min_y;
    float gmax = (grade_min_y == 0.0 && grade_max_y == 0.0) ?  50.0 : grade_max_y;
    float t = clamp((v_pos.y - gmin) / max(gmax - gmin, 0.001), 0.0, 1.0);
    col *= mix(0.96, 1.06, t);

    // snow fog rule (optional)
    float dens = fog_density;
    if (terrain_mode == 3) {
        if (wetness < 0.001) {
            dens = 0.0;      // dry snow => no fog
        } else {
            dens *= 0.5;     // wet snow => reduce fog so it is not a whiteout
        }
    }

    // clean exp2 fog toward horizon tint
    float f = fogFactor(dist, dens, v_pos.y);
    vec3  out_linear = mix(fog_color, col, f);

    // gamma out
    vec3 outCol = pow(max(out_linear, 0.0), vec3(1.0/2.2));
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
    vec4 c = texture(tex, v_tex);
    if (c.a < 0.1) discard;
    fragColor = c;
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
out vec3 v_pos;
void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_tex = in_tex;
    v_pos = in_vert;
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

// Optional controls
uniform float grade_min_y;  // default: -10 if unset
uniform float grade_max_y;  // default:  50 if unset

out vec4 fragColor;

void main() {
    vec4 s = texture(tex, v_tex);
    if (s.a < 0.1) discard;

    float w = clamp(wetness, 0.0, 1.0);
    vec3 base = s.rgb * mix(1.0, 0.88, w);

    // mild height grade
    float gmin = (grade_min_y == 0.0 && grade_max_y == 0.0) ? -10.0 : grade_min_y;
    float gmax = (grade_min_y == 0.0 && grade_max_y == 0.0) ?  50.0 : grade_max_y;
    float t = clamp((v_pos.y - gmin) / max(gmax - gmin, 0.001), 0.0, 1.0);
    base *= mix(0.97, 1.05, t);

    // clean exp2 fog toward horizon tint
    float f = exp(-fog_density * fog_density * dist * dist);
    vec3 out_linear = mix(fog_color, base, f);

    // gamma out
    vec3 col = pow(max(out_linear, 0.0), vec3(1.0/2.2));
    fragColor = vec4(col, s.a);
}
'''

rain_vertex_shader = '''
#version 330
in vec3 in_vert;
in vec4 in_color;

uniform mat4 mvp;
uniform vec3 cam_pos;
uniform float fade_distance;

out vec4 v_color;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    float dist = length(in_vert - cam_pos);
    float fade = clamp(1.0 - dist / max(fade_distance, 0.001), 0.0, 1.0);
    v_color = vec4(in_color.rgb, in_color.a * fade);
}
'''

rain_fragment_shader = '''
#version 330
in vec4 v_color;

uniform vec3 fog_color;

out vec4 fragColor;

void main() {
    if (v_color.a < 0.01) discard;
    vec3 tint = mix(fog_color, v_color.rgb, 0.35);
    fragColor = vec4(tint, v_color.a);
}
'''

puddle_vertex_shader = '''
#version 330
in vec3 in_vert;
in vec2 in_uv;
in float in_seed;

uniform mat4 mvp;

out vec2 v_uv;
out float v_seed;
out vec3 v_pos;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_uv = in_uv;
    v_seed = in_seed;
    v_pos = in_vert;
}
'''

puddle_fragment_shader = '''
#version 330
in vec2 v_uv;
in float v_seed;
in vec3 v_pos;

uniform vec3 cam_pos;
uniform vec3 fog_color;
uniform vec3 light_dir;
uniform vec3 light_color;
uniform float sky_brightness;
uniform float wetness;
uniform float rain_strength;
uniform float time;

out vec4 fragColor;

float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(dot(i, vec2(127.1, 311.7)) + v_seed * 17.0);
    float b = hash(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7)) + v_seed * 17.0);
    float c = hash(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7)) + v_seed * 17.0);
    float d = hash(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7)) + v_seed * 17.0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; ++i) {
        v += noise(p) * amp;
        p = p * 2.03 + vec2(37.2, 19.9);
        amp *= 0.5;
    }
    return v;
}

vec2 swirl(vec2 p, float t, float seed) {
    float speed = 0.8 + hash(seed * 11.7) * 0.6;
    vec2 q = p * 2.1 + vec2(t * speed, -t * 0.7);
    float sx = fbm(q + vec2(17.3, -9.1));
    float sy = fbm(q.yx + vec2(-13.7, 21.4));
    return vec2(sx, sy) * 0.45;
}

float puddle_height(vec2 uv, float t, float seed) {
    vec2 warp = swirl(uv, t * 0.6, seed);
    vec2 warped = uv * (2.8 + hash(seed * 5.3)) + warp;
    float base = fbm(warped + vec2(seed * 3.7, t * 0.8));
    float rip = sin((warped.x + warped.y) * 8.0 + t * 6.0);
    rip += sin((warped.x - warped.y) * 9.5 - t * 4.3);
    return base * 0.7 + rip * 0.08;
}

vec3 approximate_reflection(vec3 view_dir, vec3 normal, vec3 fog_color, float sky_brightness, float fresnel, float seed) {
    vec3 refl = reflect(-view_dir, normal);
    float horizon = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(vec3(0.12, 0.15, 0.20), fog_color, clamp(0.35 + sky_brightness * 0.9, 0.0, 1.0));
    vec3 ground = mix(vec3(0.07, 0.09, 0.11), fog_color, 0.75);
    vec3 env = mix(ground, sky, horizon);
    float sparkle = fbm(refl.xz * 4.3 + vec2(seed * 2.3, sky_brightness * 2.7));
    env += sparkle * 0.12 * fresnel;
    return env;
}

void main() {
    vec2 uv = v_uv;
    float seed = v_seed * 23.0 + 3.1;
    float angle = atan(uv.y, uv.x);
    float radial = length(uv);
    float outline = fbm(uv * 3.4 + vec2(cos(angle + seed), sin(angle - seed)));
    float wobble = sin(angle * 6.0 + seed * 1.9) * 0.1;
    float rim = radial + (outline - 0.5) * 0.35 + wobble;
    float mask = 1.0 - smoothstep(0.82, 1.08, rim);
    if (mask <= 0.001) discard;

    float wet = clamp(wetness, 0.0, 1.0);
    float rain = clamp(rain_strength, 0.0, 1.0);
    float t = time * (1.1 + hash(seed * 7.7) * 0.6);

    float height = puddle_height(uv, t, seed);
    float h_dx = puddle_height(uv + vec2(0.015, 0.0), t, seed) - height;
    float h_dy = puddle_height(uv + vec2(0.0, 0.015), t, seed) - height;
    float wave_amp = 0.45 + rain * 0.55;
    vec3 normal = normalize(vec3(h_dx * wave_amp, 1.0, h_dy * wave_amp));

    vec3 view_dir = normalize(cam_pos - v_pos);
    float ndotv = clamp(dot(normal, view_dir), 0.0, 1.0);
    float fresnel = pow(1.0 - ndotv, 4.0);

    vec3 env_reflect = approximate_reflection(view_dir, normal, fog_color, sky_brightness, fresnel, seed);
    float foam = fbm(uv * 5.6 + vec2(seed * 1.7, t * 2.1));
    float micro = noise(uv * 11.0 + vec2(seed * 5.3, t * 0.9)) - 0.5;

    vec3 base_tint = mix(vec3(0.05, 0.07, 0.09), fog_color, 0.4 + 0.2 * wet);
    vec3 body = base_tint + vec3(height * 0.05 + micro * 0.03);
    float depth = clamp(1.0 - rim, 0.0, 1.0);
    body = mix(body, env_reflect, 0.55 + 0.25 * (wet + rain));

    vec3 ldir = normalize(light_dir);
    float ndotl = clamp(dot(normal, ldir), 0.0, 1.0);
    vec3 diffuse = light_color * ndotl * (0.05 + 0.25 * wet);
    float spec = pow(clamp(dot(reflect(-ldir, normal), view_dir), 0.0, 1.0), 48.0);
    vec3 specular = light_color * spec * (0.25 + 0.55 * (rain + wet));

    float ring = smoothstep(0.65, 0.95, rim) * mask;
    vec3 rim_light = mix(vec3(0.18, 0.22, 0.26), fog_color, 0.55) * ring * (0.3 + 0.5 * rain);

    float sparkle = pow(max(foam, 0.0), 4.0) * (0.08 + 0.22 * rain);
    vec3 sparkling = fog_color * sparkle;

    vec3 color = body + diffuse + specular + rim_light + sparkling;
    color = mix(color, env_reflect, fresnel * (0.35 + 0.35 * wet));
    color += fresnel * fog_color * (0.06 + 0.12 * rain);
    color = clamp(color, 0.0, 1.0);

    float alpha = mask * (0.42 + 0.35 * wet + 0.45 * rain);
    alpha += fresnel * mask * (0.3 + 0.35 * wet);
    alpha += sparkle * 0.35;
    alpha = clamp(alpha, 0.0, 0.97);

    fragColor = vec4(color, alpha * (0.8 + 0.2 * depth));
}
'''

fog_sheet_vertex_shader = '''
#version 330
in vec3 in_vert;
in vec2 in_uv;
in float in_sheet;
in float in_density;

uniform mat4 mvp;

out vec2 v_uv;
out vec3 v_world;
out float v_sheet;
out float v_density;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_uv = in_uv;
    v_world = in_vert;
    v_sheet = in_sheet;
    v_density = in_density;
}
'''

fog_sheet_fragment_shader = '''
#version 330
in vec2 v_uv;
in vec3 v_world;
in float v_sheet;
in float v_density;

uniform vec3 cam_pos;
uniform vec3 fog_color;
uniform float fog_density;
uniform float rain_strength;
uniform float time;

out vec4 fragColor;

float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float hash(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash(i);
    float n100 = hash(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
}

float fbm(vec3 p) {
    float v = 0.0;
    float amp = 0.55;
    for (int i = 0; i < 4; ++i) {
        v += amp * noise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return v;
}

void main() {
    float rain = clamp(rain_strength, 0.0, 1.0);
    float base = fog_density * (1.25 + rain * 1.75);
    float seed = fract(sin(v_sheet * 37.21) * 43758.5453);

    vec3 sample_p = vec3(v_world.xz * 0.045, time * 0.25 + seed * 5.0);
    sample_p += vec3(seed * 3.1, seed * 2.3, seed * 5.7);
    float shape = fbm(sample_p);

    vec3 sweep_p = vec3(v_world.x * 0.035 + time * 0.08, v_world.z * 0.035 - time * 0.05, seed * 4.0);
    float sweep = fbm(sweep_p);

    vec3 layer_p = vec3(v_world.xz * 0.022, time * 0.12 + seed * 2.7);
    float layers = fbm(layer_p);

    float band = abs(fract((v_world.x + v_world.z) * 0.03 + time * 0.12 + seed) - 0.5);
    band = 1.0 - smoothstep(0.08, 0.42, band);

    float height_fade = smoothstep(1.25, -0.05, v_uv.y * 1.4);
    float edge = smoothstep(0.0, 0.2, v_uv.x) * (1.0 - smoothstep(0.8, 1.0, v_uv.x));
    edge *= smoothstep(0.0, 0.25, v_uv.y) * (1.0 - smoothstep(0.9, 1.0, v_uv.y));

    float density = base * v_density;
    density *= (0.55 + 0.55 * shape);
    density *= (0.55 + 0.50 * sweep);
    density *= mix(0.7, 1.35, layers);
    density *= (0.45 + 0.55 * band);
    density *= (0.35 + 0.65 * height_fade);
    float camera_dist = length(vec2(v_world.x - cam_pos.x, v_world.z - cam_pos.z));
    float proximity = clamp(camera_dist / 90.0, 0.0, 1.0);
    density *= mix(0.6 + 0.2 * rain, 1.0, proximity);
    density *= edge;

    float alpha = clamp(density * 2.4, 0.0, 0.88);
    if (alpha < 0.01) {
        discard;
    }

    vec3 tint = mix(fog_color, fog_color * (0.75 + 0.35 * layers), 0.55);
    tint += fog_color * (0.06 + 0.10 * rain) * band;

    fragColor = vec4(clamp(tint, 0.0, 1.0), alpha);
}
'''

def create_shaders(ctx):
    try:
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        prog_lit = ctx.program(vertex_shader=vertex_shader_lit, fragment_shader=fragment_shader_lit)
        prog2d = ctx.program(vertex_shader=vertex_shader_2d, fragment_shader=fragment_shader_2d)
        prog_tex = ctx.program(vertex_shader=vertex_shader_tex, fragment_shader=fragment_shader_tex)
        prog_rain = ctx.program(vertex_shader=rain_vertex_shader, fragment_shader=rain_fragment_shader)
        prog_puddle = ctx.program(vertex_shader=puddle_vertex_shader, fragment_shader=puddle_fragment_shader)
        prog_fog = ctx.program(vertex_shader=fog_sheet_vertex_shader, fragment_shader=fog_sheet_fragment_shader)
        print("Shaders compiled successfully")
        return prog, prog2d, prog_lit, prog_tex, prog_rain, prog_puddle, prog_fog
    except Exception as e:
        print(f"Shader compilation error: {e}")
