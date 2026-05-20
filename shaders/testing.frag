#version 410

uniform vec2 u_resolution; // viewport size in pixels (width, height)
uniform int u_frame; // frame increment counter
uniform vec2 u_mouse;
uniform vec2 u_cameraRot;
uniform vec3 u_cameraPos;
uniform float u_time;
uniform vec3 u_tempColor;
uniform float u_focusDist;
uniform float u_aperture;

// Accumulation Texture
uniform sampler2D u_pass1;

// Textures
uniform sampler2D u_ground;
uniform sampler2D u_ground_normal;
uniform sampler2D u_ground_disp;
uniform sampler2D u_ground_roughness;

uniform sampler2D u_onyx;
uniform sampler2D u_onyx_roughness;
uniform sampler2D u_onyx_displacement;

uniform sampler2D u_tile;
uniform sampler2D u_tile_roughness;
uniform sampler2D u_tile_displacement;

uniform sampler2D u_main;
uniform int u_spf; // 16, [1, 64]
out vec4 fragColor;

// a pixel value multiplier of light before tone mapping and sRGB
const float c_exposure = 0.5f;
const float KEY_SPACE = 32.5 / 256.0;
const float c_rayPosNormalNudge = 0.01f;

vec3 LessThan(vec3 f, float value) {
  return vec3((f.x < value) ? 1.0f : 0.0f, (f.y < value) ? 1.0f : 0.0f,
    (f.z < value) ? 1.0f : 0.0f);
}

vec3 LinearToSRGB(vec3 rgb) {
  rgb = clamp(rgb, 0.0f, 1.0f);

  return mix(pow(rgb, vec3(1.0f / 2.4f)) * 1.055f - 0.055f, rgb * 12.92f,
    LessThan(rgb, 0.0031308f));
}

vec3 SRGBToLinear(vec3 rgb) {
  rgb = clamp(rgb, 0.0f, 1.0f);

  return mix(pow(((rgb + 0.055f) / 1.055f), vec3(2.4f)), rgb / 12.92f,
    LessThan(rgb, 0.04045f));
}

// ACES tone mapping curve fit to go from HDR to LDR
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x) {
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

float FresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incident,
  float f0, float f90) {
  // Schlick aproximation
  float r0 = (n1 - n2) / (n1 + n2);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if (n1 > n2) {
    float n = n1 / n2;
    float sinT2 = n * n * (1.0 - cosX * cosX);
    // Total internal reflection
    if (sinT2 > 1.0)
      return f90;
    cosX = sqrt(1.0 - sinT2);
  }
  float x = 1.0 - cosX;
  float ret = r0 + (1.0 - r0) * x * x * x * x * x;

  // adjust reflect multiplier for object reflectivity
  return mix(f0, f90, ret);
}

const int MAX_STEPS = 192;
const float EPSILON = 0.001;
const float MAX_DIST = 500.0;
const float c_FOVDegrees = 90.0f;
const float c_pi = 3.14159265359f;
const float c_twopi = 2.0f * c_pi;
// the farthest we look for path tracing ray hits
const float c_superFar = 10000.0f;
const int c_numBounces = 8;
const int SHADOW_MAX_STEPS = 96;
const vec3 c_sunPos = vec3(15.0, 25.0, 10.0);
const float c_sunRadius = 5.0;
const float c_sunEmissionStrength = 50.0;

struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 target;
};

struct Material {
  vec3 albedo; // Base color (surface reflections/F0)
  float roughness; // 0.0 = smooth, 1.0 = frosted
  float metallic;
  vec3 emissive;

  // transparent materials
  float transmission;
  float IOR;
  vec3 absorption;
};

struct SDF {
  float distance;
  float id;
};

struct SRayHitInfo {
  bool fromInside;
  float dist;
  vec3 normal;
  Material material;
};

float opUnion(float a, float b) {
  return min(a, b);
}

SDF opUnionID(SDF res1, SDF res2) {
  if (res1.distance < res2.distance) {
    return res1;
  } else {
    return res2;
  }
}

float RandomValue(inout float state) {
  // Normalize the state to avoid precision issues with large numbers in sin()
  state = fract(state * 0.1031);
  float result = fract(sin(state * 12.9898 + 78.233) * 43758.5453123);
  state += 1.0;
  return result;
}

float RandomFloat01(inout float state) {
  return RandomValue(state);
}

// Random value in normal distribution (with mean=0 and sd=1)
float RandomValueNormalDistribution(inout float state) {
  float u1 = RandomValue(state); // Placeholder for float-based RNG
  float u2 = RandomValue(state); // Placeholder for float-based RNG

  // Safeguard against log(0) which is undefined/infinity
  // We use a tiny offset (epsilon)
  u1 = max(u1, 1e-7);

  float theta = 2.0 * 3.14159265359 * u2;
  float rho = sqrt(-2.0 * log(u1));
  return rho * cos(theta);
}

// Calculate a random direction
vec3 RandomDirection(inout float state) {
  // Thanks to https://math.stackexchange.com/a/1585996
  float x = RandomValueNormalDistribution(state);
  float y = RandomValueNormalDistribution(state);
  float z = RandomValueNormalDistribution(state);
  return normalize(vec3(x, y, z));
}

// Cook torrance things
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
  // Disney/Epic Games roughness remapping
  float a = roughness * roughness;

  // Convert our random numbers into spherical coordinates (phi and theta)
  // based on the exact GGX distribution curve
  float phi = 2.0 * c_pi * Xi.x;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // Convert spherical coordinates to tangent-space vector
  vec3 H;
  H.x = cos(phi) * sinTheta;
  H.y = sin(phi) * sinTheta;
  H.z = cosTheta;

  // We need to rotate this tangent-space vector to align with our actual
  // surface Normal We create a Coordinate System (Tangent, Bitangent, Normal)
  vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, N));
  vec3 bitangent = cross(N, tangent);

  // Transform H from tangent space to world space
  vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

float dot2(vec3 v) {
  return dot(v, v);
}

float sdSphere(vec3 p, float r) {
  return length(p) - r;
}

vec3 RandomUnitVector(inout float state) {
  float z = RandomFloat01(state) * 2.0f - 1.0f;
  float a = RandomFloat01(state) * c_twopi;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);
  return vec3(x, y, z);
}

vec3 CosineSampleHemisphere(vec2 Xi, vec3 N) {
  float phi = c_twopi * Xi.x;
  float cosTheta = sqrt(1.0 - Xi.y);
  float sinTheta = sqrt(Xi.y);

  vec3 H;
  H.x = cos(phi) * sinTheta;
  H.y = sin(phi) * sinTheta;
  H.z = cosTheta;

  vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, N));
  vec3 bitangent = cross(N, tangent);
  vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(sampleVec);
}

float GGXGeometrySchlick(float NdotV, float roughness) {
  float k = pow(roughness + 1.0, 2.0) * 0.125;
  return NdotV / max(NdotV * (1.0 - k) + k, 1e-6);
}

float GGXGeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  return GGXGeometrySchlick(NdotV, roughness) * GGXGeometrySchlick(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float sdPlane(vec3 p, vec3 n, float h) {
  // n must be normalized
  return dot(p, n) + h;
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

Material GetZeroedMaterial() {
  Material ret;
  ret.albedo = vec3(0.0);
  ret.roughness = 1.0;
  ret.metallic = 0.0;
  ret.emissive = vec3(0.0);
  ret.transmission = 0.0;
  ret.IOR = 1.0;
  ret.absorption = vec3(0.0);
  return ret;
}

vec3 triPlanar(sampler2D tex, vec3 p, vec3 normal) {
  normal = abs(normal);
  normal = pow(normal, vec3(5.0));
  normal /= normal.x + normal.y + normal.z;
  return (texture(tex, p.xy * 0.5 + 0.5) * normal.z +
    texture(tex, p.xz * 0.5 + 0.5) * normal.y +
    texture(tex, p.yz * 0.5 + 0.5) * normal.x)
  .rgb;
}

vec3 triPlanarNormal(sampler2D tex, vec3 p, vec3 geomNormal) {
  // 1. Calculate the blend weights based on the macro normal
  vec3 blend = abs(geomNormal);
  blend = pow(blend, vec3(5.0));
  blend /= (blend.x + blend.y + blend.z);

  // 2. Sample the normal map 3 times and unpack to [-1, 1] vectors
  vec3 tX = texture(tex, p.yz * 0.5).rgb * 2.0 - 1.0;
  vec3 tY = texture(tex, p.xz * 0.5).rgb * 2.0 - 1.0;
  vec3 tZ = texture(tex, p.xy * 0.5).rgb * 2.0 - 1.0;

  // 3. Prevent inverted bumps on the "back" sides of the SDF.
  // We flip the tangent (X) of the normal map based on the facing direction.
  tX.x *= sign(geomNormal.x);
  tY.x *= sign(geomNormal.y); // Flipped X instead of Y
  tZ.x *= sign(geomNormal.z); // Flipped X instead of Z

  tX = vec3(tX.xy + geomNormal.zy, abs(tX.z) * geomNormal.x);
  tY = vec3(tY.xy + geomNormal.xz, abs(tY.z) * geomNormal.y);
  tZ = vec3(tZ.xy + geomNormal.xy, abs(tZ.z) * geomNormal.z);

  // 5. Swizzle the vectors to align with world space based on UV mapping!
  // p.yz (U=Y, V=Z) -> TexX=WorldY, TexY=WorldZ
  vec3 nX = vec3(tX.z, tX.x, tX.y);
  // p.xz (U=X, V=Z) -> TexX=WorldX, TexY=WorldZ
  vec3 nY = vec3(tY.x, tY.z, tY.y);
  // p.xy (U=X, V=Y) -> TexX=WorldX, TexY=WorldY
  vec3 nZ = vec3(tZ.x, tZ.y, tZ.z);

  // 6. Blend and return the final bent normal
  return normalize(nX * blend.x + nY * blend.y + nZ * blend.z);
}

SDF map(vec3 p) {
    // 1. Light Source
    SDF sun;
    sun.distance = sdSphere(p - c_sunPos, c_sunRadius);
    sun.id = 0.0;

    // 2. The Ground (40x40 platform, top surface at Y = 0.0)
    SDF ground;
    ground.distance = sdBox(p - vec3(0.0, -1.0, 0.0), vec3(10.0, 1.0, 10.0));
    ground.id = 4.0; 

    SDF result = opUnionID(sun, ground);

    // 3. The Centerpiece (Pedestal + Sphere)
    // Pedestal: half-height is 0.5. Center at Y = 0.5 so it sits on the ground.
    SDF pedestal;
    pedestal.distance = sdBox(p - vec3(0.0, 0.5, 0.0), vec3(1.2, 0.5, 1.2));
    pedestal.id = 1.0; 
    result = opUnionID(result, pedestal);

    // Center Sphere: Radius 1.2. Pedestal top is at Y = 1.0. 
    // Center it at Y = 2.2 so it perfectly rests on the pedestal.
    SDF centerSphere;
    centerSphere.distance = sdSphere(p - vec3(0.0, 2.2, 0.0), 1.2);
    centerSphere.id = 10.0; // Make it Gold/Metal so it stands out!
    result = opUnionID(result, centerSphere);

    // 4. Original "Chess Pieces"
    SDF box2;
    box2.distance = sdBox(p - vec3(-3.5, 1.0, 1.5), vec3(1.0, 1.0, 1.0));
    box2.id = 1.0; 
    result = opUnionID(result, box2);

    SDF box3;
    box3.distance = sdBox(p - vec3(3.5, 2.0, -1.0), vec3(0.6, 2.0, 0.6));
    box3.id = 3.0; // Changed to Red
    result = opUnionID(result, box3);

    SDF box4;
    box4.distance = sdBox(p - vec3(1.5, 0.5, 4.0), vec3(0.5, 0.5, 0.5));
    box4.id = 2.0; 
    result = opUnionID(result, box4);

    SDF box5;
    box5.distance = sdBox(p - vec3(-2.0, 0.8, -4.0), vec3(0.8, 0.8, 0.8));
    box5.id = 8.0; 
    result = opUnionID(result, box5);

    SDF box6;
    box6.distance = sdBox(p - vec3(4.5, 1.2, -5.0), vec3(1.2, 1.2, 1.2));
    box6.id = 1.0; 
    result = opUnionID(result, box6);

    // 5. NEW Extra Boxes to fill out the scene
    
    // Tall pillar in the deep back-left
    SDF box7;
    box7.distance = sdBox(p - vec3(-6.0, 2.5, -5.0), vec3(0.5, 2.5, 0.5));
    box7.id = 1.0;
    result = opUnionID(result, box7);

    // Wide, low barricade box in the front
    SDF box8;
    box8.distance = sdBox(p - vec3(-1.0, 0.4, 6.5), vec3(2.0, 0.4, 0.6));
    box8.id = 8.0; // Rough Red
    result = opUnionID(result, box8);

    // Tiny cluster box next to box2
    SDF box9;
    box9.distance = sdBox(p - vec3(-5.0, 0.6, 2.5), vec3(0.6, 0.6, 0.6));
    box9.id = 2.0; // Greenish
    result = opUnionID(result, box9);

    // Medium box far right foreground
    SDF box10;
    box10.distance = sdBox(p - vec3(6.5, 1.5, 3.5), vec3(0.8, 1.5, 0.8));
    box10.id = 1.0;
    result = opUnionID(result, box10);

    // Deep background accent
    SDF box11;
    box11.distance = sdBox(p - vec3(1.0, 1.0, -7.0), vec3(1.0, 1.0, 1.0));
    box11.id = 7.0; // Gold Metal
    result = opUnionID(result, box11);

    return result;
}

vec3 getNormal(vec3 p) {
  vec2 e = vec2(EPSILON, 0.0);
  return normalize(vec3(map(p + e.xyy).distance - map(p - e.xyy).distance,
      map(p + e.yxy).distance - map(p - e.yxy).distance,
      map(p + e.yyx).distance - map(p - e.yyx).distance));
}

SDF rayMarch(vec3 ro, vec3 rd) {
  SDF hit, object;
  float t = 0.0;
  for (int i = 0; i < MAX_STEPS; i++) {
    vec3 p = ro + rd * t;
    hit = map(p); // Distance to nearest object
    // Keep a small minimum step to avoid stalls near surfaces while marching faster in open space.
    float stepDist = max(abs(hit.distance) * 0.9, EPSILON * 0.5);
    t += stepDist;
    object.distance = hit.distance;
    object.id = hit.id;
    if (abs(hit.distance) < EPSILON || t > MAX_DIST)
      break;
  }
  object.distance = t;
  object.id = hit.id;
  return object;
}

vec3 applyNormalMap(vec3 geomNormal, vec3 texNormal) {
  // Convert texture RGB from [0.0, 1.0] to direction vectors [-1.0, 1.0]
  texNormal = texNormal * 2.0 - 1.0;

  // Build the TBN (Tangent, Bitangent, Normal) coordinate system
  vec3 up =
    abs(geomNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tangent = normalize(cross(up, geomNormal));
  vec3 bitangent = cross(geomNormal, tangent);

  // Rotate the texture's bump direction to match the geometry's surface
  vec3 finalNormal = tangent * texNormal.x + bitangent * texNormal.y +
      geomNormal * texNormal.z;

  return normalize(finalNormal);
}

Material getMaterial(SDF object, vec3 p, inout vec3 normal) {
  Material material = GetZeroedMaterial();
  switch (int(object.id)) {
    // white light
    case 0:
    // material.emissive = vec3(1, 0.7529422167760779, 0.5775804404296506)
    // * 20.0f; material.emissive = vec3(1.0) * 20.0f;
    vec3 color = SRGBToLinear(u_tempColor);
    material.emissive = color * 50.0f;
    material.albedo = vec3(0);
    break;
    //
    case 1:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.1, 0.7, 0.1);
    break;
    // white
    case 2:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.7, 0.7, 0.7);
    break;
    // red
    case 3:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.7, 0.1, 0.1);
    break;
    // back wall
    case 4:
    material.emissive = vec3(0.0, 0.0, 0.0);

    // Scale controls the size of the squares. 
    // 2.0 makes large squares, 20.0 makes very tiny squares.
    float scale = 2.0; 

    // Add the floored X and Z coordinates, then modulo 2
    float checker = mod(floor(p.x * scale) + floor(p.z * scale), 2.0);

    // Mix between light grey (0.7) and dark grey (0.2)
    material.albedo = mix(vec3(0.7f, 0.7f, 0.7f), vec3(0.2f, 0.2f, 0.2f), checker);
    break;
    // left wall
    case 5:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.7f, 0.1f, 0.1f);
    break;
    // right wall
    case 6:
    material.emissive = vec3(0.0, 0.0, 0.0);
    material.albedo = vec3(0.1f, 0.7f, 0.1f);
    break;
    case 7:
    material.albedo = SRGBToLinear(triPlanar(u_onyx, p, normal).rgb);
    material.metallic = 0.0;
    material.roughness = triPlanar(u_onyx_roughness, p, normal).r;
    break;
    case 8:
    material.albedo = vec3(0.8, 0.1, 0.1); // Red
    material.metallic = 0.0; // Not a metal
    material.roughness = 1.0; // Quite rough
    break;
    case 9:
    // vec3 texColor = triPlanar(u_ground, p, normal).rgb;
    // float texRough = triPlanar(u_ground_roughness, p, normal).r;
    vec3 texColor = texture(u_ground, p.xz).rgb;
    float texRough = texture(u_ground_roughness, p.xz).r;

    material.albedo = SRGBToLinear(texColor);
    material.roughness = texRough;
    material.metallic = 0.0;

    // normal = triPlanarNormal(u_ground_normal, p, normal);

    break;
    // glass
    case 10:
    material.albedo = vec3(0.0, 0.0, 0.0);
    material.metallic = 0.0;
    material.transmission = 1.0;
    material.IOR = 1.3;
    material.roughness = 0.02;
    material.absorption = vec3(0.0);
    break;
    case 11:
    // material.albedo = SRGBToLinear(triPlanar(u_tile, p, normal).rgb);
    // material.roughness = triPlanar(u_tile_roughness, p, normal).r;
    material.albedo = SRGBToLinear(texture(u_tile, p.xy).rgb);
    material.roughness = texture(u_tile_roughness, p.xy).r;
    material.metallic = 0.0;
    break;
    case 12:
    material.albedo = vec3(1.0, 1.0, 0.0);
    break;
    default:
    material.albedo = vec3(1.0, 0.0, 1.0); // bright magenta for unhandled ids
    break;
  }
  return material;
}

void getCurrentHit(vec3 rayOrigin, vec3 rayDirection,
  inout SRayHitInfo hitInfo) {
  SDF object = rayMarch(rayOrigin, rayDirection);
  if (object.distance < MAX_DIST) {
    hitInfo.dist = object.distance;
    hitInfo.normal = getNormal(rayOrigin + rayDirection * object.distance);
    hitInfo.material = getMaterial(
        object, rayOrigin + rayDirection * object.distance, hitInfo.normal);

    if (dot(rayDirection, hitInfo.normal) > 0.0) {
      hitInfo.normal = -hitInfo.normal;
      hitInfo.fromInside = true;
    }
  } else {
    hitInfo.dist = c_superFar;
  }
}

bool isVisibleToLight(vec3 origin, vec3 dir, float maxDist) {
  float t = EPSILON * 4.0;
  for (int i = 0; i < SHADOW_MAX_STEPS; ++i) {
    if (t >= maxDist) {
      return true;
    }
    vec3 p = origin + dir * t;
    SDF h = map(p);
    if (h.distance < EPSILON) {
      return h.id == 0.0;
    }
    t += max(abs(h.distance) * 0.95, EPSILON * 0.5);
  }
  return true;
}

vec3 sampleDirectSun(vec3 hitPos, vec3 normal, Material mat, inout float rngState) {
  if (mat.transmission > 0.0) {
    return vec3(0.0);
  }

  vec3 lightDirRandom = RandomUnitVector(rngState);
  vec3 lightPoint = c_sunPos + lightDirRandom * c_sunRadius;
  vec3 toLight = lightPoint - hitPos;
  float dist2 = dot(toLight, toLight);
  if (dist2 <= 1e-5) {
    return vec3(0.0);
  }

  float dist = sqrt(dist2);
  vec3 wi = toLight / dist;
  float ndotl = max(dot(normal, wi), 0.0);
  if (ndotl <= 0.0) {
    return vec3(0.0);
  }

  if (!isVisibleToLight(hitPos + normal * c_rayPosNormalNudge, wi, dist - EPSILON * 4.0)) {
    return vec3(0.0);
  }

  vec3 lightNormal = normalize(lightPoint - c_sunPos);
  float lightCos = max(dot(lightNormal, -wi), 0.0);
  if (lightCos <= 0.0) {
    return vec3(0.0);
  }

  float area = 4.0 * c_pi * c_sunRadius * c_sunRadius;
  float pdfA = 1.0 / area;
  float pdfW = pdfA * dist2 / max(lightCos, 1e-4);
  if (pdfW <= 0.0) {
    return vec3(0.0);
  }

  vec3 diffuseColor = mat.albedo * (1.0 - mat.metallic);
  vec3 f = diffuseColor * (1.0 / c_pi);
  vec3 Le = SRGBToLinear(u_tempColor) * c_sunEmissionStrength;

  return Le * f * ndotl / pdfW;
}

vec3 getColorForRay(in vec3 startRayPos, in vec3 startRayDir,
  inout float rngState) {
  // initialize
  vec3 ret = vec3(0.0f, 0.0f, 0.0f);
  vec3 throughput = vec3(1.0f, 1.0f, 1.0f);
  vec3 rayPos = startRayPos;
  vec3 rayDir = startRayDir;
  bool prevWasDelta = true;

  for (int bounceIndex = 0; bounceIndex <= c_numBounces; ++bounceIndex) {
    // shoot a ray out into the world
    SRayHitInfo hitInfo;
    hitInfo.material = GetZeroedMaterial();
    hitInfo.dist = c_superFar;
    hitInfo.fromInside = false;
    getCurrentHit(rayPos, rayDir, hitInfo);

    // if the ray missed, we are done
    if (hitInfo.dist == c_superFar) {
      break;
    }

    if (prevWasDelta || bounceIndex == 0) {
      ret += throughput * hitInfo.material.emissive;
    }

    if (length(hitInfo.material.emissive) < 1e-6 &&
      hitInfo.material.transmission <= 0.0) {
      vec3 hitPosDirect = rayPos + rayDir * hitInfo.dist;
      ret += throughput * sampleDirectSun(hitPosDirect, hitInfo.normal, hitInfo.material,
          rngState);
    }

    // Beers law for volume absorption
    if (hitInfo.fromInside)
      throughput *= exp(-hitInfo.material.absorption * hitInfo.dist);

    float n1 = hitInfo.fromInside ? hitInfo.material.IOR : 1.0f;
    float n2 = hitInfo.fromInside ? 1.0f : hitInfo.material.IOR;
    float eta = n1 / n2;

    // Base reflection chance (0.04 is the standard F0 for most non-metals)
    float fresnelChance =
      FresnelReflectAmount(n1, n2, hitInfo.normal, rayDir, 0.04f, 1.0f);
    // Metals force 100% reflection. Dielectrics reflect based on Fresnel.
    float P_reflect = mix(fresnelChance, 1.0f, hitInfo.material.metallic);
    // Transmission (glass) only happens if the ray didn't reflect.
    float P_transmit = hitInfo.material.transmission * (1.0f - P_reflect);
    // Diffuse happens if it didn't reflect and didn't transmit.
    float P_diffuse = max(1.0f - (P_reflect + P_transmit), 0.0f);

    float branchSum = max(P_reflect + P_transmit + P_diffuse, 1e-4);
    P_reflect /= branchSum;
    P_transmit /= branchSum;
    P_diffuse /= branchSum;

    float roll = RandomFloat01(rngState);
    vec3 hitPos = rayPos + rayDir * hitInfo.dist;

    if (roll < P_reflect) {
      // PATH A: SPECULAR REFLECTION
      vec3 viewDir = normalize(-rayDir);
      float roughness = max(hitInfo.material.roughness, 0.02);
      vec3 H = ImportanceSampleGGX(
          vec2(RandomFloat01(rngState), RandomFloat01(rngState)),
          hitInfo.normal, roughness);
      if (dot(rayDir, H) > 0.0) {
        H = -H;
      }
      rayDir = reflect(rayDir, H);
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      float NdotV = max(dot(hitInfo.normal, viewDir), 1e-4);
      float NdotH = max(dot(hitInfo.normal, H), 1e-4);
      float VdotH = max(dot(viewDir, H), 1e-4);
      vec3 F0 = mix(vec3(0.04f), hitInfo.material.albedo,
          hitInfo.material.metallic);
      vec3 F = FresnelSchlick(VdotH, F0);
      float G = GGXGeometrySmith(hitInfo.normal, viewDir, rayDir, roughness);

      throughput *= F * G * VdotH / max(NdotV * NdotH, 1e-4);
      throughput /= P_reflect;
      prevWasDelta = roughness <= 0.02;
    } else if (roll < P_reflect + P_transmit) {
      // PATH B: TRANSMISSION (GLASS/WATER)
      float roughness = max(hitInfo.material.roughness, 0.02);
      vec3 H = ImportanceSampleGGX(
          vec2(RandomFloat01(rngState), RandomFloat01(rngState)),
          hitInfo.normal, roughness);
      if (dot(rayDir, H) > 0.0) {
        H = -H;
      }

      vec3 refractedDir = refract(rayDir, H, eta);
      if (dot(refractedDir, refractedDir) < 1e-6) {
        rayDir = reflect(rayDir, H);
        throughput /= P_reflect;
      } else {
        rayDir = normalize(refractedDir);
        throughput /= P_transmit;
      }

      // Nudge the ray *into* the object
      rayPos = hitPos - hitInfo.normal * c_rayPosNormalNudge;

      // Note: We don't apply surface color here. Color comes from volume
      // absorption.
      prevWasDelta = roughness <= 0.02;
    } else {
      // PATH C: DIFFUSE
      rayDir = CosineSampleHemisphere(
          vec2(RandomFloat01(rngState), RandomFloat01(rngState)),
          hitInfo.normal);
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      // Metals have no diffuse color.
      vec3 diffuseColor =
        hitInfo.material.albedo * (1.0f - hitInfo.material.metallic);
      throughput *= diffuseColor;
      throughput /= P_diffuse;
      prevWasDelta = false;
    }

    // 4. Russian Roulette (Path Termination)
    if (bounceIndex >= 3) {
      float p = max(throughput.r, max(throughput.g, throughput.b));
      p = clamp(p, 0.05f, 0.95f);
      if (RandomFloat01(rngState) > p) {
        break;
      }
      throughput *= 1.0f / p;
    }
  }

  return ret;
}

void pR(inout vec2 p, float a) {
  p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

mat3 getCam(vec3 ro, vec3 lookAt) {
  vec3 camF = normalize(vec3(lookAt - ro));
  vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
  vec3 camU = cross(camF, camR);
  return mat3(camR, camU, camF);
}

void applyRotation(inout vec3 ro) {
  // Pitch (up/down) around the XZ plane
  pR(ro.yz, u_cameraRot.y);
  // Yaw (left/right) around the vertical axis
  pR(ro.xz, u_cameraRot.x);
}

vec2 RandomPointInCircle(inout float state) {
    float angle = RandomFloat01(state) * c_twopi; // Placeholder for float-based RNG
    float radius = sqrt(RandomFloat01(state)); // Placeholder for float-based RNG
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
  // RNG setup
  float state = float(gl_FragCoord.x) * 1973.0 +
    float(gl_FragCoord.y) * 9277.0 + float(u_frame) * 26699.0 + 1.0;
  
  // DEBUG: Test if RNG is working
  vec3 debug_rng = vec3(RandomFloat01(state), RandomFloat01(state), RandomFloat01(state));
  
  vec3 accumulated_color = vec3(0.0);
  int samples_per_frame = u_spf;
  float focusDist = u_focusDist;  // How far away the perfectly in-focus plane is
  float aperture = u_aperture;  // The radius of the lens (higher = blurrier background)

  for (int i = 0; i < samples_per_frame; i++) {
    vec2 jitter = vec2(RandomFloat01(state), RandomFloat01(state)) - 0.5; // Placeholder for float-based RNG
    vec2 uv = (gl_FragCoord.xy + jitter) / u_resolution.xy;
    vec2 pixelTarget2D = (uv * 2.0f) - 1.0f;
    float aspectRatio = u_resolution.x / u_resolution.y;

    // calculate the camera distance
    float cameraDistance = 1.0f / tan(c_FOVDegrees * 0.5f * c_pi / 180.0f);

    // Ray ray;
    // // 1. The origin is strictly controlled by WASD
    // ray.origin = u_cameraPos;

// 2. Base ray direction (pointing down the -Z axis)
    vec3 rayDir = normalize(
        vec3(pixelTarget2D.x * aspectRatio, pixelTarget2D.y, -cameraDistance));

    // Rotate the ray itself based on mouse movement
    pR(rayDir.yz, u_cameraRot.y); // Pitch up/down
    pR(rayDir.xz, u_cameraRot.x); // Yaw left/right
    
    // 3. Calculate camera's Forward, Right and Up vectors in world space 
    // by applying the exact same rotations to base axis vectors
    vec3 camForward = vec3(0.0, 0.0, -1.0);
    pR(camForward.yz, u_cameraRot.y);
    pR(camForward.xz, u_cameraRot.x);

    vec3 camRight = vec3(1.0, 0.0, 0.0);
    pR(camRight.yz, u_cameraRot.y);
    pR(camRight.xz, u_cameraRot.x);

    vec3 camUp = vec3(0.0, 1.0, 0.0);
    pR(camUp.yz, u_cameraRot.y);
    pR(camUp.xz, u_cameraRot.x);

    // --- THE FIX ---
    // Adjust the distance so it forms a flat plane, not a sphere
    float trueFocusDist = focusDist / dot(rayDir, camForward);
    vec3 focal_point = u_cameraPos + rayDir * trueFocusDist;

    // 4. Jitter the ray origin within the aperture disk
    vec2 lensJitter = RandomPointInCircle(state) * aperture;
    
    Ray ray;
    ray.origin = u_cameraPos + camRight * lensJitter.x + camUp * lensJitter.y;

    // 5. Point the jittered ray at the original focal point
    ray.direction = normalize(focal_point - ray.origin);

    accumulated_color += getColorForRay(ray.origin, ray.direction, state);
  }

  vec3 current_average = accumulated_color / float(samples_per_frame);

  // average the frames together
  vec4 lastFrameColor = texture(u_pass1, gl_FragCoord.xy / u_resolution.xy);

  // NEW MATH:
  // We need to tell the blend that we just added 64 samples, not 1.
  // If we use u_frame, it's much safer:
  float blend = 1.0 / float(u_frame + 1);

  vec3 color = mix(lastFrameColor.rgb, current_average, blend);
  
  // DEBUG: Output accumulated color directly (clamped so we can see if anything is being computed)
  color = clamp(accumulated_color, 0.0, 1.0) * 0.1;

  // show the result - use a solid 1.0 for Alpha so Main pass can see it!
  fragColor = vec4(color, 1.0);
}
