#version 410

// ============================================================================
// NEE PATH-TRACER TEMPLATE
// ----------------------------------------------------------------------------
// Self-contained starting point for a new Next-Event-Estimation scene. The
// generic path-tracing machinery below is copied from the CORRECT reference
// shader, ct-newScene1.frag. Do NOT copy scene1-test.frag / scene2.frag — they
// have glass-critical NEE bugs (see CLAUDE.md).
//
// To make a new scene you should ONLY need to edit:
//   1. The light constants + the `#define NEE_LIGHT_*` switch (just below).
//   2. The two SCENE-SPECIFIC sections: `map()` and `getMaterial()`.
// Everything else (RNG, BRDF, the bounce loop) is the reusable engine.
//
// Three NEE invariants the engine preserves — keep them if you touch the loop:
//   * Delta exclusion: NEE is skipped when `roughness <= 0.02` (smooth glass OR
//     smooth mirror/metal). NEE on a delta BRDF is invalid.
//   * Double-count guard: `prevWasDelta` gates the implicit emissive hit so
//     direct light is counted exactly once (via NEE on rough surfaces, via the
//     implicit hit after a delta bounce).
//   * TIR / horizon breaks in the transmission branch: refract() returns the
//     zero vector on total internal reflection — the loop breaks instead of
//     marching a degenerate (0,0,0) ray (the cause of black glass edges).
//
// LIGHT-ID COUPLING (easy to get wrong): the shadow-ray test `isVisibleToLight`
// treats a hit as "the light" when `h.id == 0.0`. So the emissive light object
// in `map()` MUST use id 0. If you change the light's id, update both places.
// ============================================================================

// ---- Choose the light model for this scene --------------------------------
// Exactly one of these. SPHERE = distant spherical sun. RECT = ceiling area light.
#define NEE_LIGHT_SPHERE
// #define NEE_LIGHT_RECT

uniform vec2 u_resolution; // viewport size in pixels (width, height)
uniform int u_frame;       // frame increment counter
uniform vec2 u_mouse;
uniform vec2 u_cameraRot;
uniform vec3 u_cameraPos;
uniform float u_time;
uniform vec3 u_tempColor;
uniform float u_focusDist;
uniform float u_aperture;

uniform sampler2D u_pass1;
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

// ---- Light constants (edit per scene) -------------------------------------
// Spherical sun (used when NEE_LIGHT_SPHERE is defined):
const vec3 c_sunPos = vec3(100.0, 100.0, 0.0);
const float c_sunRadius = 10.0;
const float c_sunEmissionStrength = 500.0;
// Ceiling rectangular area light (used when NEE_LIGHT_RECT is defined):
const vec3 c_lightPos = vec3(0.0, 20.0, 0.0); // center of the light
const vec2 c_lightSize = vec2(8.0, 8.0);      // HALF extents in x and z
const float c_lightEmissionStrength = 50.0;

struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 target;
};

struct Material {
  vec3 albedo;     // Base color (surface reflections/F0)
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

float opUnion(float a, float b) { return min(a, b); }

SDF opUnionID(SDF res1, SDF res2) {
  if (res1.distance < res2.distance) {
    return res1;
  } else {
    return res2;
  }
}

uint NextRandom(inout uint state) {
  state = state * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  uint result = (word >> 22u) ^ word;
  return result;
}

float RandomValue(inout uint state) {
  return float(NextRandom(state)) / 4294967295.0; // 2^32 - 1
}

float RandomValueNormalDistribution(inout uint state) {
  float u1 = RandomValue(state);
  float u2 = RandomValue(state);
  u1 = max(u1, 1e-7);
  float theta = 2.0 * 3.14159265359 * u2;
  float rho = sqrt(-2.0 * log(u1));
  return rho * cos(theta);
}

vec3 RandomDirection(inout uint state) {
  float x = RandomValueNormalDistribution(state);
  float y = RandomValueNormalDistribution(state);
  float z = RandomValueNormalDistribution(state);
  return normalize(vec3(x, y, z));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
  float a = roughness * roughness;
  float phi = 2.0 * c_pi * Xi.x;
  float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

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

vec3 RandomUnitVector(inout uint state) {
  float z = RandomValue(state) * 2.0f - 1.0f;
  float a = RandomValue(state) * c_twopi;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);
  return vec3(x, y, z);
}

// ---- SDF primitive helpers (subset of Mercury hg_sdf) ----------------------
float dot2(vec3 v) { return dot(v, v); }
float vmax(vec2 v) { return max(v.x, v.y); }
float vmax(vec3 v) { return max(max(v.x, v.y), v.z); }
float vmax(vec4 v) { return max(max(v.x, v.y), max(v.z, v.w)); }

void pR(inout vec2 p, float a) {
    p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

float sgn(float x) {
    if (x < 0.0) return -1.0;
    if (x > 0.0) return 1.0;
    return 0.0;
}

vec2 sgn(vec2 v) {
    return vec2((v.x < 0.0) ? -1.0 : 1.0, (v.y < 0.0) ? -1.0 : 1.0);
}

float fBox2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, vec2(0))) + vmax(min(d, vec2(0)));
}

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdPlane(vec3 p, vec3 n, float h) { return dot(p, n) + h; }

float pMod1(inout float p, float size) {
    float halfsize = size * 0.5;
    float c = floor((p + halfsize) / size);
    p = mod(p + halfsize, size) - halfsize;
    return c;
}

float sdCappedCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float pMirror(inout float p, float dist) {
    float s = sgn(p);
    p = abs(p) - dist;
    return s;
}

vec2 pMirrorOctant(inout vec2 p, vec2 dist) {
    vec2 s = sgn(p);
    pMirror(p.x, dist.x);
    pMirror(p.y, dist.y);
    if (p.y > p.x)
        p.xy = p.yx;
    return s;
}

SDF opDifferenceID(SDF res1, SDF res2) {
    return (res1.distance > -res2.distance) ? res1 : SDF(-res2.distance, res2.id);
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

// ============================================================================
// SCENE-SPECIFIC: EDIT THIS  (geometry)
// ----------------------------------------------------------------------------
// Build your scene here. Return the closest SDF (distance + id). The id is what
// getMaterial() switches on. The emissive LIGHT object MUST keep id 0 (see the
// LIGHT-ID COUPLING note at the top).
// ============================================================================
SDF map(vec3 p) {
  // --- The light (id 0). Its shape matches the chosen NEE estimator. ---
  SDF light;
#ifdef NEE_LIGHT_RECT
  // Thin ceiling box centered at c_lightPos, half-extents (x,z) = c_lightSize.
  light.distance = sdBox(p - c_lightPos, vec3(c_lightSize.x, 0.05, c_lightSize.y));
#else
  // Distant spherical sun.
  light.distance = sdSphere(p - c_sunPos, c_sunRadius);
#endif
  light.id = 0.0;

  // --- Ground plane (id 1, diffuse). ---
  SDF ground;
  ground.distance = sdPlane(p, vec3(0.0, 1.0, 0.0), 14.0); // plane at y = -14
  ground.id = 1.0;

  // --- Demo object: a smooth glass sphere (id 2) to exercise the TIR path. ---
  SDF sphere;
  sphere.distance = sdSphere(p, 5.0);
  sphere.id = 2.0;

  SDF result = opUnionID(ground, sphere);
  result = opUnionID(result, light);
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
    hit = map(p);
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

// ============================================================================
// SCENE-SPECIFIC: EDIT THIS  (materials)
// ----------------------------------------------------------------------------
// One case per id used in map(). Keep id 0 emissive (the light). Set
// roughness <= 0.02 for smooth (delta) glass/mirror surfaces.
// ============================================================================
Material getMaterial(SDF object, vec3 p) {
  Material material = GetZeroedMaterial();
  switch (int(object.id)) {
  case 0: // the light — emission must match the NEE estimator's Le.
#ifdef NEE_LIGHT_RECT
    material.emissive = SRGBToLinear(u_tempColor) * c_lightEmissionStrength;
#else
    material.emissive = SRGBToLinear(u_tempColor) * c_sunEmissionStrength;
#endif
    material.albedo = vec3(0.0);
    break;
  case 1: // ground — diffuse grey
    material.albedo = vec3(0.7, 0.7, 0.7);
    material.roughness = 1.0;
    break;
  case 2: // demo — smooth glass (delta surface; exercises refraction + TIR)
    material.albedo = vec3(0.0);
    material.metallic = 0.0;
    material.transmission = 1.0;
    material.IOR = 1.5;
    material.roughness = 0.02;
    material.absorption = vec3(0.0);
    break;
  default:
    material.albedo = vec3(1.0, 0.0, 1.0); // magenta = unassigned id
    break;
  }
  return material;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0; // Path tracing / IBL k-value
    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Shadow ray: returns true if `dir` reaches the light unobstructed. A surface
// counts as "the light" when its id is 0 (keep map()'s light at id 0).
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
  return false;
}

// Full Cook-Torrance BRDF (diffuse + GGX specular), shared by both NEE
// estimators. Matches the reference NEE BRDF in ct-newScene1.frag.
vec3 evalCookTorranceBRDF(vec3 N, vec3 V, vec3 L, Material mat) {
  vec3 H = normalize(V + L);

  float NdotV = max(dot(N, V), 1e-4);
  float NdotL = max(dot(N, L), 1e-4);
  float NdotH = max(dot(N, H), 0.0);
  float HdotV = max(dot(H, V), 0.0);

  // 1. Diffuse component
  vec3 diffuseColor = mat.albedo * (1.0 - mat.metallic);
  vec3 f_diffuse = diffuseColor * (1.0 / c_pi);

  // 2. Specular component (GGX)
  float a = mat.roughness * mat.roughness;
  float a2 = a * a;

  float denomD = (NdotH * NdotH * (a2 - 1.0) + 1.0);
  float D = a2 / (c_pi * denomD * denomD);

  float k_direct = ((mat.roughness + 1.0) * (mat.roughness + 1.0)) / 8.0;
  float ggx1 = NdotL / (NdotL * (1.0 - k_direct) + k_direct);
  float ggx2 = NdotV / (NdotV * (1.0 - k_direct) + k_direct);
  float G = ggx1 * ggx2;

  vec3 F0 = mix(vec3(0.04), mat.albedo, mat.metallic);
  vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);

  vec3 f_specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.001);

  // Energy conservation
  vec3 kS = F;
  vec3 kD = vec3(1.0) - kS;
  kD *= 1.0 - mat.metallic; // pure metals have no diffuse

  return (kD * f_diffuse) + f_specular;
}

// NEE estimator for a spherical sun. Returns Le * f * NdotL / pdfW.
vec3 sampleDirectSun(vec3 hitPos, vec3 normal, vec3 viewDir, Material mat,
                     inout uint rngState) {
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

  if (!isVisibleToLight(hitPos + normal * c_rayPosNormalNudge, wi,
                        dist - EPSILON * 4.0)) {
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

  vec3 f = evalCookTorranceBRDF(normal, viewDir, wi, mat);
  float NdotL = max(dot(normal, wi), 1e-4);
  vec3 Le = SRGBToLinear(u_tempColor) * c_sunEmissionStrength;

  return Le * f * NdotL / pdfW;
}

// NEE estimator for a ceiling rectangular area light. Returns Le * f * NdotL / pdfW.
vec3 sampleRectLight(vec3 hitPos, vec3 normal, vec3 viewDir, Material mat,
                     inout uint rngState) {
  if (mat.transmission > 0.0) {
    return vec3(0.0);
  }

  // 1. Pick a random point on the bottom face of the rectangular light.
  float rX = RandomValue(rngState) * 2.0 - 1.0;
  float rZ = RandomValue(rngState) * 2.0 - 1.0;

  // Offset Y to the bottom face of the light box (matches map()'s half-height).
  vec3 lightPoint = c_lightPos + vec3(rX * c_lightSize.x, -0.05, rZ * c_lightSize.y);

  // The normal of a ceiling light points straight down.
  vec3 lightNormal = vec3(0.0, -1.0, 0.0);

  vec3 toLight = lightPoint - hitPos;
  float dist2 = dot(toLight, toLight);
  if (dist2 <= 1e-5) {
    return vec3(0.0);
  }

  float dist = sqrt(dist2);
  vec3 wi = toLight / dist;

  if (!isVisibleToLight(hitPos + normal * c_rayPosNormalNudge, wi,
                        dist - EPSILON * 4.0)) {
    return vec3(0.0);
  }

  float lightCos = max(dot(lightNormal, -wi), 0.0);
  if (lightCos <= 0.0) {
    return vec3(0.0); // hit the side/back of the light
  }

  // Area of the rect = (2 * half_x) * (2 * half_z); convert area PDF to solid angle.
  float area = (2.0 * c_lightSize.x) * (2.0 * c_lightSize.y);
  float pdfA = 1.0 / area;
  float pdfW = pdfA * dist2 / max(lightCos, 1e-4);
  if (pdfW <= 0.0) {
    return vec3(0.0);
  }

  vec3 f = evalCookTorranceBRDF(normal, viewDir, wi, mat);
  float NdotL = max(dot(normal, wi), 1e-4);
  vec3 Le = SRGBToLinear(u_tempColor) * c_lightEmissionStrength;

  return Le * f * NdotL / pdfW;
}

// Dispatch to the estimator chosen by the NEE_LIGHT_* switch at the top.
vec3 sampleDirectLight(vec3 hitPos, vec3 normal, vec3 viewDir, Material mat,
                       inout uint rngState) {
#ifdef NEE_LIGHT_RECT
  return sampleRectLight(hitPos, normal, viewDir, mat, rngState);
#else
  return sampleDirectSun(hitPos, normal, viewDir, mat, rngState);
#endif
}

void getCurrentHit(vec3 rayOrigin, vec3 rayDirection,
                   inout SRayHitInfo hitInfo) {
  SDF object = rayMarch(rayOrigin, rayDirection);
  if (object.distance < MAX_DIST) {
    hitInfo.dist = object.distance;
    hitInfo.normal = getNormal(rayOrigin + rayDirection * object.distance);
    hitInfo.material = getMaterial(object, rayOrigin + rayDirection * object.distance);

    if (dot(rayDirection, hitInfo.normal) > 0.0) {
      hitInfo.normal = -hitInfo.normal;
      hitInfo.fromInside = true;
    }
  } else {
    hitInfo.dist = c_superFar;
  }
}

// ============================================================================
// PATH TRACER LOOP — the NEE engine. Generic; preserve the three invariants
// described in the header if you modify it.
// ============================================================================
vec3 getColorForRay(in vec3 startRayPos, in vec3 startRayDir,
                    inout uint rngState) {
  vec3 ret = vec3(0.0f, 0.0f, 0.0f);
  vec3 throughput = vec3(1.0f, 1.0f, 1.0f);
  vec3 rayPos = startRayPos;
  vec3 rayDir = startRayDir;
  bool prevWasDelta = true;

  for (int bounceIndex = 0; bounceIndex <= c_numBounces; ++bounceIndex) {
    SRayHitInfo hitInfo;
    hitInfo.material = GetZeroedMaterial();
    hitInfo.dist = c_superFar;
    hitInfo.fromInside = false;
    getCurrentHit(rayPos, rayDir, hitInfo);

    if (hitInfo.dist == c_superFar) {
      break;
    }

    // 1. Implicit Light Hit (only when not already counted via NEE)
    if (prevWasDelta || bounceIndex == 0) {
      ret += throughput * hitInfo.material.emissive;
    }

    // Evaluate if this is a delta material ONCE
    bool isDelta = hitInfo.material.roughness <= 0.02;

    // 2. Next Event Estimation (Direct Lighting) — skip on lights and deltas
    if (length(hitInfo.material.emissive) < 1e-6 && !isDelta) {
      vec3 hitPosDirect = rayPos + rayDir * hitInfo.dist;
      ret += throughput * sampleDirectLight(hitPosDirect, hitInfo.normal, -rayDir, hitInfo.material, rngState);
    }

    // 3. Beer's Law for volume absorption
    if (hitInfo.fromInside) {
      throughput *= exp(-hitInfo.material.absorption * hitInfo.dist);
    }

    // 4. Calculate Path Probabilities
    float n1 = hitInfo.fromInside ? hitInfo.material.IOR : 1.0f;
    float n2 = hitInfo.fromInside ? 1.0f : hitInfo.material.IOR;
    float eta = n1 / n2;

    float fresnelChance = FresnelReflectAmount(n1, n2, hitInfo.normal, rayDir, 0.04f, 1.0f);
    float P_reflect = mix(fresnelChance, 1.0f, hitInfo.material.metallic);
    float P_transmit = hitInfo.material.transmission * (1.0f - P_reflect);
    float P_diffuse = 1.0f - (P_reflect + P_transmit);

    float roll = RandomValue(rngState);
    vec3 hitPos = rayPos + rayDir * hitInfo.dist;

    // 5. Route the path
    if (roll < P_reflect) {
      // PATH A: SPECULAR REFLECTION
      vec3 reflectionColor = mix(vec3(1.0f), hitInfo.material.albedo, hitInfo.material.metallic);

      if (isDelta) {
        // Bypass GGX math for smooth surfaces (Prevents 3000x energy spikes)
        rayDir = reflect(rayDir, hitInfo.normal);
        throughput *= reflectionColor;
      } else {
        vec3 V = -rayDir;
        vec3 H = ImportanceSampleGGX(vec2(RandomValue(rngState), RandomValue(rngState)), hitInfo.normal, hitInfo.material.roughness);
        rayDir = reflect(rayDir, H);

        // Horizon Check: Kill ray if the microfacet reflected it INSIDE the object
        if (dot(rayDir, hitInfo.normal) <= 0.0) break;

        vec3 L = rayDir;
        float NdotV = max(dot(hitInfo.normal, V), 1e-4);
        float NdotL = max(dot(hitInfo.normal, L), 1e-4);
        float NdotH = max(dot(hitInfo.normal, H), 1e-4);
        float VdotH = max(dot(V, H), 1e-4);
        float G = GeometrySmith(hitInfo.normal, V, L, hitInfo.material.roughness);
        float ggxWeight = (G * VdotH) / (NdotV * NdotH);

        throughput *= reflectionColor * ggxWeight;
      }

      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;
      prevWasDelta = isDelta;

    } else if (roll < P_reflect + P_transmit) {
      // PATH B: TRANSMISSION (GLASS/WATER)
      if (isDelta) {
        // Bypass GGX math
        rayDir = refract(rayDir, hitInfo.normal, eta);
      } else {
        vec3 H = ImportanceSampleGGX(vec2(RandomValue(rngState), RandomValue(rngState)), hitInfo.normal, hitInfo.material.roughness);
        rayDir = refract(rayDir, H, eta);
      }

      if (length(rayDir) < 0.01f) {
        break; // Total Internal Reflection
      }

      // Horizon Check: Kill ray if microfacet refracted it back OUT of the object
      if (dot(rayDir, hitInfo.normal) >= 0.0) break;

      rayPos = hitPos - hitInfo.normal * c_rayPosNormalNudge;
      prevWasDelta = isDelta;

    } else {
      // PATH C: DIFFUSE
      rayDir = normalize(hitInfo.normal + RandomUnitVector(rngState));
      rayPos = hitPos + hitInfo.normal * c_rayPosNormalNudge;

      throughput *= hitInfo.material.albedo;
      prevWasDelta = false;
    }

    // 6. Russian Roulette (Path Termination)
    float p = min(max(throughput.r, max(throughput.g, throughput.b)), 1.0f);

    if (RandomValue(rngState) > p) {
      break;
    }

    if (p > 0.0) {
       throughput *= 1.0f / p;
    }
  }

  return ret;
}

mat3 getCam(vec3 ro, vec3 lookAt) {
  vec3 camF = normalize(vec3(lookAt - ro));
  vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
  vec3 camU = cross(camF, camR);
  return mat3(camR, camU, camF);
}

void applyRotation(inout vec3 ro) {
  pR(ro.yz, u_cameraRot.y);
  pR(ro.xz, u_cameraRot.x);
}

vec2 RandomPointInCircle(inout uint state) {
    float angle = RandomValue(state) * c_twopi;
    float radius = sqrt(RandomValue(state));
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
  uint state =
    uint(uint(gl_FragCoord.x) * uint(1973) +
        uint(gl_FragCoord.y) * uint(9277) + uint(u_frame) * uint(26699)) |
      uint(1);

  state = NextRandom(state);
  state = NextRandom(state);
  vec3 accumulated_color = vec3(0.0);
  int samples_per_frame = u_spf;
  float focusDist = u_focusDist;
  float aperture = u_aperture;

  for (int i = 0; i < samples_per_frame; i++) {
    vec2 jitter = vec2(RandomValue(state), RandomValue(state)) - 0.5;
    vec2 uv = (gl_FragCoord.xy + jitter) / u_resolution.xy;
    vec2 pixelTarget2D = (uv * 2.0f) - 1.0f;
    float aspectRatio = u_resolution.x / u_resolution.y;

    float cameraDistance = 1.0f / tan(c_FOVDegrees * 0.5f * c_pi / 180.0f);

    vec3 rayDir = normalize(
        vec3(pixelTarget2D.x * aspectRatio, pixelTarget2D.y, -cameraDistance));

    pR(rayDir.yz, u_cameraRot.y);
    pR(rayDir.xz, u_cameraRot.x);

    vec3 focal_point = u_cameraPos + rayDir * focusDist;

    vec3 camRight = vec3(1.0, 0.0, 0.0);
    pR(camRight.yz, u_cameraRot.y);
    pR(camRight.xz, u_cameraRot.x);

    vec3 camUp = vec3(0.0, 1.0, 0.0);
    pR(camUp.yz, u_cameraRot.y);
    pR(camUp.xz, u_cameraRot.x);

    vec2 lensJitter = RandomPointInCircle(state) * aperture;

    Ray ray;
    ray.origin = u_cameraPos + camRight * lensJitter.x + camUp * lensJitter.y;
    ray.direction = normalize(focal_point - ray.origin);

    accumulated_color += getColorForRay(ray.origin, ray.direction, state);
  }

  vec3 current_average = accumulated_color / float(samples_per_frame);
  vec4 lastFrameColor = texture(u_pass1, gl_FragCoord.xy / u_resolution.xy);
  float blend = 1.0 / float(u_frame + 1);
  vec3 color = mix(lastFrameColor.rgb, current_average, blend);

  fragColor = vec4(color, 1.0);
}
